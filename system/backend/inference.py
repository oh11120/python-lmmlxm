from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import nibabel as nib
import numpy as np

from brats_seg.data.preprocess import denoise_volume, n4_bias_field_correction, zscore_nonzero
from brats_seg.models.factory import build_model


class InferenceEngine:
    def __init__(
        self,
        backend: str,
        model_name: str,
        ckpt_path: str,
        onnx_path: str,
        trt_engine_path: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.ckpt_path = ckpt_path
        self.onnx_path = onnx_path
        self.trt_engine_path = trt_engine_path
        self._pt_model = None
        self._ort_session = None
        self._trt_ctx = None

    def _load_pytorch(self):
        if self._pt_model is not None:
            return
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(self.model_name, num_classes=4).to(device)
        path = Path(self.ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.ckpt_path}")
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        self._pt_model = (model, device)

    def _load_onnx(self):
        if self._ort_session is not None:
            return
        import onnxruntime as ort

        path = Path(self.onnx_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX file not found: {self.onnx_path}")
        self._ort_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def _load_tensorrt(self):
        if self._trt_ctx is not None:
            return
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except Exception as exc:
            raise ImportError("TensorRT backend requires `tensorrt` and `pycuda`.") from exc

        path = Path(self.trt_engine_path or "")
        if not path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {path}")

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        self._trt_ctx = {"trt": trt, "cuda": cuda, "engine": engine, "context": context}

    def _infer_tensorrt(self, x: np.ndarray) -> np.ndarray:
        self._load_tensorrt()
        trt = self._trt_ctx["trt"]
        cuda = self._trt_ctx["cuda"]
        engine = self._trt_ctx["engine"]
        context = self._trt_ctx["context"]

        input_name = engine.get_tensor_name(0)
        output_name = engine.get_tensor_name(1)

        context.set_input_shape(input_name, x.shape)
        out_shape = tuple(context.get_tensor_shape(output_name))
        out = np.empty(out_shape, dtype=np.float32)

        d_input = cuda.mem_alloc(x.nbytes)
        d_output = cuda.mem_alloc(out.nbytes)

        stream = cuda.Stream()
        cuda.memcpy_htod_async(d_input, x, stream)
        context.set_tensor_address(input_name, int(d_input))
        context.set_tensor_address(output_name, int(d_output))
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(out, d_output, stream)
        stream.synchronize()
        return out

    @staticmethod
    def _load_modalities(case_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        required = ["flair", "t1", "t1ce", "t2"]
        raw_vols = {}
        flair_affine = None
        for mod in required:
            matches = list(case_dir.glob(f"*_{mod}.nii"))
            if not matches:
                raise FileNotFoundError(f"Missing modality {mod} in {case_dir}")
            nii = nib.load(str(matches[0]))
            raw_vols[mod] = np.asarray(nii.dataobj, dtype=np.float32)
            if mod == "flair":
                flair_affine = nii.affine

        n4_mask = (raw_vols["t1ce"] != 0).astype(np.uint8)
        vols = []
        for mod in required:
            vol = denoise_volume(raw_vols[mod], median_size=3, gaussian_sigma=1.0)
            vol = n4_bias_field_correction(vol, mask=n4_mask, convergence_threshold=1e-6, fail_if_unavailable=True)
            vol = zscore_nonzero(vol)
            vols.append(vol)

        image = np.stack(vols, axis=0)  # [C,D,H,W]
        return image, flair_affine

    @staticmethod
    def _compute_starts(length: int, patch: int, stride: int):
        if length <= patch:
            return [0]
        starts = list(range(0, max(length - patch, 0) + 1, stride))
        if starts[-1] != length - patch:
            starts.append(length - patch)
        return starts

    def _predict_patch(self, x: np.ndarray) -> np.ndarray:
        if self.backend == "onnx":
            self._load_onnx()
            inp_name = self._ort_session.get_inputs()[0].name
            return self._ort_session.run(None, {inp_name: x})[0]
        if self.backend == "tensorrt":
            return self._infer_tensorrt(x)

        self._load_pytorch()
        import torch

        model, device = self._pt_model
        with torch.no_grad():
            return model(torch.from_numpy(x).to(device)).cpu().numpy()

    @staticmethod
    def _extract_patch_with_pad(image: np.ndarray, z0: int, y0: int, x0: int, patch_size):
        c, d, h, w = image.shape
        pd, ph, pw = patch_size
        z1, y1, x1 = min(d, z0 + pd), min(h, y0 + ph), min(w, x0 + pw)
        patch = image[:, z0:z1, y0:y1, x0:x1]

        out = np.zeros((c, pd, ph, pw), dtype=image.dtype)
        rz, ry, rx = z1 - z0, y1 - y0, x1 - x0
        out[:, :rz, :ry, :rx] = patch
        return out, (z1, y1, x1), (rz, ry, rx)

    def _infer_full_volume(self, image: np.ndarray, patch_size=(128, 128, 128), overlap: float = 0.5) -> np.ndarray:
        # image: [C,D,H,W]
        c, d, h, w = image.shape
        pd, ph, pw = patch_size
        sd = max(1, int(pd * (1.0 - overlap)))
        sh = max(1, int(ph * (1.0 - overlap)))
        sw = max(1, int(pw * (1.0 - overlap)))

        z_starts = self._compute_starts(d, pd, sd)
        y_starts = self._compute_starts(h, ph, sh)
        x_starts = self._compute_starts(w, pw, sw)

        first, _, _ = self._extract_patch_with_pad(image, z_starts[0], y_starts[0], x_starts[0], patch_size)
        first_logits = self._predict_patch(first[None, ...].astype(np.float32))
        num_classes = first_logits.shape[1]

        logits_acc = np.zeros((1, num_classes, d, h, w), dtype=np.float32)
        weight_acc = np.zeros((1, 1, d, h, w), dtype=np.float32)

        for z0 in z_starts:
            for y0 in y_starts:
                for x0 in x_starts:
                    patch, (z1, y1, x1), (rz, ry, rx) = self._extract_patch_with_pad(image, z0, y0, x0, patch_size)
                    logits = self._predict_patch(patch[None, ...].astype(np.float32))
                    logits_acc[:, :, z0:z1, y0:y1, x0:x1] += logits[:, :, :rz, :ry, :rx]
                    weight_acc[:, :, z0:z1, y0:y1, x0:x1] += 1.0

        return logits_acc / np.clip(weight_acc, 1e-6, None)

    @staticmethod
    def _to_brats_labels(mask: np.ndarray) -> np.ndarray:
        out = mask.copy().astype(np.uint8)
        out[out == 3] = 4
        return out

    def infer_case(self, case_dir: str, output_seg_path: str) -> Dict[str, str]:
        image, affine = self._load_modalities(Path(case_dir))
        patch_size = (128, 128, 128)
        if self.model_name.lower() == "unetr":
            patch_size = (128, 128, 256)
        logits = self._infer_full_volume(image=image, patch_size=patch_size, overlap=0.5)
        pred = np.argmax(logits, axis=1)[0].astype(np.uint8)
        pred = self._to_brats_labels(pred)
        nib.save(nib.Nifti1Image(pred, affine), output_seg_path)
        return {"seg_path": output_seg_path}
