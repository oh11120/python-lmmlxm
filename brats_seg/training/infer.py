from __future__ import annotations

from typing import Tuple

import torch


def _compute_starts(length: int, patch: int, stride: int):
    if length <= patch:
        return [0]
    starts = list(range(0, max(length - patch, 0) + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def _extract_patch_with_pad(
    image: torch.Tensor,
    z0: int,
    y0: int,
    x0: int,
    patch_size: Tuple[int, int, int],
) -> torch.Tensor:
    # image: [B,C,D,H,W]
    _, _, d, h, w = image.shape
    pd, ph, pw = patch_size

    z1, y1, x1 = min(d, z0 + pd), min(h, y0 + ph), min(w, x0 + pw)
    patch = image[:, :, z0:z1, y0:y1, x0:x1]

    pad_d = pd - (z1 - z0)
    pad_h = ph - (y1 - y0)
    pad_w = pw - (x1 - x0)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        # F.pad order for 5D: (Wl, Wr, Hl, Hr, Dl, Dr)
        patch = torch.nn.functional.pad(patch, (0, pad_w, 0, pad_h, 0, pad_d), mode="constant", value=0.0)
    return patch


def sliding_window_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch_size: Tuple[int, int, int],
    overlap: float = 0.5,
) -> torch.Tensor:
    """Run full-volume inference by stitching overlapped patch logits.

    image shape: [B, C, D, H, W], current implementation assumes B=1.
    """
    if image.ndim != 5:
        raise ValueError(f"Expected [B,C,D,H,W], got shape={tuple(image.shape)}")
    if image.shape[0] != 1:
        raise ValueError("Only batch size 1 is supported in sliding window inference")

    _, c, d, h, w = image.shape
    pd, ph, pw = patch_size
    sd = max(1, int(pd * (1.0 - overlap)))
    sh = max(1, int(ph * (1.0 - overlap)))
    sw = max(1, int(pw * (1.0 - overlap)))

    z_starts = _compute_starts(d, pd, sd)
    y_starts = _compute_starts(h, ph, sh)
    x_starts = _compute_starts(w, pw, sw)

    first_patch = _extract_patch_with_pad(image, z_starts[0], y_starts[0], x_starts[0], patch_size)
    first_logits = model(first_patch)
    num_classes = first_logits.shape[1]

    logits_acc = torch.zeros((1, num_classes, d, h, w), device=image.device, dtype=first_logits.dtype)
    weight_acc = torch.zeros((1, 1, d, h, w), device=image.device, dtype=first_logits.dtype)

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                patch = _extract_patch_with_pad(image, z0, y0, x0, patch_size)
                logits = model(patch)
                z1, y1, x1 = min(d, z0 + pd), min(h, y0 + ph), min(w, x0 + pw)
                rz, ry, rx = z1 - z0, y1 - y0, x1 - x0
                logits_acc[:, :, z0:z1, y0:y1, x0:x1] += logits[:, :, :rz, :ry, :rx]
                weight_acc[:, :, z0:z1, y0:y1, x0:x1] += 1.0

    return logits_acc / torch.clamp_min(weight_acc, 1e-6)
