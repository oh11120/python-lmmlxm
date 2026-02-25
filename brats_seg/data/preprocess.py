from __future__ import annotations

from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def load_nifti(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).dataobj, dtype=np.float32)


def zscore_nonzero(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mask = volume != 0
    if not np.any(mask):
        return volume
    vals = volume[mask]
    mean = vals.mean()
    std = vals.std()
    std = max(float(std), eps)
    out = volume.copy()
    out[mask] = (out[mask] - mean) / std
    return out


def denoise_volume(volume: np.ndarray, median_size: int = 3, gaussian_sigma: float = 1.0) -> np.ndarray:
    out = median_filter(volume, size=median_size)
    out = gaussian_filter(out, sigma=gaussian_sigma)
    return out.astype(np.float32)


def n4_bias_field_correction(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    convergence_threshold: float = 1e-6,
    fail_if_unavailable: bool = False,
) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ImportError:
        if fail_if_unavailable:
            raise
        return volume

    img = sitk.GetImageFromArray(volume.astype(np.float32))
    if mask is None:
        mask = (volume != 0).astype(np.uint8)
    mask_img = sitk.GetImageFromArray(mask.astype(np.uint8))
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetConvergenceThreshold(convergence_threshold)
    corrected = n4.Execute(img, mask_img)
    return sitk.GetArrayFromImage(corrected).astype(np.float32)


def remap_brats_labels(label: np.ndarray) -> np.ndarray:
    # BraTS uses {0,1,2,4}; map to contiguous {0,1,2,3}
    out = label.astype(np.int64)
    out[out == 4] = 3
    return out


def random_crop_3d(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: Tuple[int, int, int],
    rng: np.random.Generator,
    tumor_bias: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    # image shape: [C, D, H, W], label shape: [D, H, W]
    _, d, h, w = image.shape
    pd, ph, pw = patch_size

    if d <= pd or h <= ph or w <= pw:
        return _crop_or_pad_to_size(image, label, patch_size)

    use_tumor_center = rng.random() < tumor_bias and np.any(label > 0)
    if use_tumor_center:
        tumor_pos = np.argwhere(label > 0)
        cz, cy, cx = tumor_pos[rng.integers(0, len(tumor_pos))]
        z0 = int(np.clip(cz - pd // 2, 0, d - pd))
        y0 = int(np.clip(cy - ph // 2, 0, h - ph))
        x0 = int(np.clip(cx - pw // 2, 0, w - pw))
    else:
        z0 = int(rng.integers(0, d - pd + 1))
        y0 = int(rng.integers(0, h - ph + 1))
        x0 = int(rng.integers(0, w - pw + 1))

    return (
        image[:, z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw],
        label[z0 : z0 + pd, y0 : y0 + ph, x0 : x0 + pw],
    )


def _crop_or_pad_to_size(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    pd, ph, pw = patch_size
    c, d, h, w = image.shape
    out_img = np.zeros((c, pd, ph, pw), dtype=image.dtype)
    out_lbl = np.zeros((pd, ph, pw), dtype=label.dtype)

    cd, ch, cw = min(d, pd), min(h, ph), min(w, pw)
    z0_i = max(0, (d - cd) // 2)
    y0_i = max(0, (h - ch) // 2)
    x0_i = max(0, (w - cw) // 2)
    z0_o = max(0, (pd - cd) // 2)
    y0_o = max(0, (ph - ch) // 2)
    x0_o = max(0, (pw - cw) // 2)

    out_img[:, z0_o : z0_o + cd, y0_o : y0_o + ch, x0_o : x0_o + cw] = image[
        :, z0_i : z0_i + cd, y0_i : y0_i + ch, x0_i : x0_i + cw
    ]
    out_lbl[z0_o : z0_o + cd, y0_o : y0_o + ch, x0_o : x0_o + cw] = label[
        z0_i : z0_i + cd, y0_i : y0_i + ch, x0_i : x0_i + cw
    ]
    return out_img, out_lbl
