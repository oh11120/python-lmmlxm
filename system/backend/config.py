from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    storage_root: Path = Path("system/storage")
    upload_root: Path = Path("system/storage/uploads")
    result_root: Path = Path("system/storage/results")
    report_root: Path = Path("system/storage/reports")
    task_root: Path = Path("system/storage/tasks")
    api_token: str = os.getenv("BRATS_API_TOKEN", "change-me-token")
    default_backend: str = os.getenv("INFER_BACKEND", "pytorch")
    model_name: str = os.getenv("MODEL_NAME", "improved_unet3d")
    model_ckpt: str = os.getenv("MODEL_CKPT", "artifacts/exp_main/best_model.pt")
    onnx_path: str = os.getenv("ONNX_PATH", "artifacts/onnx/improved_unet3d.onnx")
    trt_engine_path: str = os.getenv("TRT_ENGINE_PATH", "artifacts/trt/improved_unet3d_fp16.engine")


settings = Settings()
for p in [settings.storage_root, settings.upload_root, settings.result_root, settings.report_root, settings.task_root]:
    p.mkdir(parents=True, exist_ok=True)
