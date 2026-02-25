from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate, zoom


def random_flip(image: np.ndarray, label: np.ndarray, rng: np.random.Generator):
    # image [C,D,H,W], label [D,H,W]
    for axis in (1, 2, 3):
        if rng.random() < 0.5:
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis - 1).copy()
    return image, label


def random_rotate_scale(
    image: np.ndarray,
    label: np.ndarray,
    rng: np.random.Generator,
    max_angle: float = 10.0,
    scale_range: Tuple[float, float] = (0.9, 1.1),
):
    angle = float(rng.uniform(-max_angle, max_angle))
    scale = float(rng.uniform(scale_range[0], scale_range[1]))

    # Rotate in-plane (H, W), keep D unchanged.
    rotated = np.stack(
        [rotate(ch, angle=angle, axes=(1, 2), reshape=False, order=1, mode="nearest") for ch in image],
        axis=0,
    )
    label_rot = rotate(label, angle=angle, axes=(1, 2), reshape=False, order=0, mode="nearest")

    if abs(scale - 1.0) < 1e-3:
        return rotated.astype(np.float32), label_rot.astype(np.int64)

    zf = (1.0, scale, scale)
    scaled = np.stack([zoom(ch, zoom=zf, order=1) for ch in rotated], axis=0)
    label_scaled = zoom(label_rot, zoom=zf, order=0)

    # center crop/pad back to original size
    return _match_shape(scaled, image.shape), _match_shape(label_scaled, label.shape)


def _match_shape(arr: np.ndarray, target_shape):
    out = arr
    for axis, target in enumerate(target_shape):
        cur = out.shape[axis]
        if cur > target:
            start = (cur - target) // 2
            slicer = [slice(None)] * out.ndim
            slicer[axis] = slice(start, start + target)
            out = out[tuple(slicer)]
        elif cur < target:
            pad_before = (target - cur) // 2
            pad_after = target - cur - pad_before
            pad = [(0, 0)] * out.ndim
            pad[axis] = (pad_before, pad_after)
            out = np.pad(out, pad, mode="constant")
    return out


def random_elastic_deform(
    image: np.ndarray,
    label: np.ndarray,
    rng: np.random.Generator,
    deformation_coeff: float = 0.05,
    prob: float = 0.3,
):
    if rng.random() > prob:
        return image, label

    # Report-style elastic deformation using GridSample with coefficient 0.05.
    img_t = torch.from_numpy(image).unsqueeze(0).float()  # [1,C,D,H,W]
    lbl_t = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()  # [1,1,D,H,W]

    _, _, d, h, w = img_t.shape
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, d),
        torch.linspace(-1.0, 1.0, h),
        torch.linspace(-1.0, 1.0, w),
        indexing="ij",
    )
    base_grid = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0)  # [1,D,H,W,3]
    disp = torch.from_numpy((rng.random((1, d, h, w, 3)).astype(np.float32) * 2.0 - 1.0))
    grid = torch.clamp(base_grid + disp * deformation_coeff, -1.0, 1.0)

    out_img = F.grid_sample(img_t, grid, mode="bilinear", padding_mode="border", align_corners=True)
    out_lbl = F.grid_sample(lbl_t, grid, mode="nearest", padding_mode="border", align_corners=True)
    return out_img.squeeze(0).numpy().astype(np.float32), out_lbl.squeeze(0).squeeze(0).numpy().astype(np.int64)
