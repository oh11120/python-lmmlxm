from __future__ import annotations

from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np


def _normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    nz = img[img > 0]
    if nz.size == 0:
        return np.zeros_like(img, dtype=np.uint8)
    lo, hi = np.percentile(nz, [1, 99])
    img = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
    return (img * 255).astype(np.uint8)


def _overlay(gray: np.ndarray, seg: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    color_map = {
        1: np.array([255, 80, 80], dtype=np.float32),
        2: np.array([80, 255, 80], dtype=np.float32),
        4: np.array([80, 170, 255], dtype=np.float32),
    }
    out = rgb.copy()
    for cls, color in color_map.items():
        m = seg == cls
        out[m] = (1 - alpha) * out[m] + alpha * color
    return out.clip(0, 255).astype(np.uint8)


def _mask_rgba(seg: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*seg.shape, 4), dtype=np.uint8)
    color_map = {
        1: (255, 80, 80, 255),
        2: (80, 255, 80, 255),
        4: (80, 170, 255, 255),
    }
    for cls, color in color_map.items():
        rgba[seg == cls] = np.array(color, dtype=np.uint8)
    return rgba


def generate_preview_images(flair_path: str, seg_path: str, out_dir: str) -> Dict[str, str]:
    from PIL import Image

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    flair = np.asarray(nib.load(flair_path).dataobj, dtype=np.float32)
    seg = np.asarray(nib.load(seg_path).dataobj, dtype=np.uint8)

    idx = np.argwhere(seg > 0)
    if len(idx) > 0:
        zc, yc, xc = idx.mean(axis=0).astype(int)
    else:
        zc, yc, xc = np.array(seg.shape) // 2

    views = {
        "axial": (flair[:, :, zc], seg[:, :, zc]),
        "sagittal": (flair[xc, :, :], seg[xc, :, :]),
        "coronal": (flair[:, yc, :], seg[:, yc, :]),
    }

    output = {}
    for name, (img, lab) in views.items():
        g = _normalize(img)
        o = _overlay(g, lab)
        base = out / f"{name}_base.png"
        mask = out / f"{name}_mask.png"
        over = out / f"{name}.png"
        Image.fromarray(np.stack([g, g, g], axis=-1)).save(base)
        Image.fromarray(_mask_rgba(lab), mode="RGBA").save(mask)
        Image.fromarray(o).save(over)
        output[name] = str(over)
        output[f"{name}_base"] = str(base)
        output[f"{name}_mask"] = str(mask)
    return output
