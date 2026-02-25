from __future__ import annotations

from typing import Dict

import torch


def compute_segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 4) -> Dict[str, float]:
    pred = torch.argmax(logits, dim=1)
    metrics = {}

    dice_vals = []
    iou_vals = []
    sen_vals = []
    spe_vals = []

    for c in range(1, num_classes):
        p = pred == c
        t = target == c

        tp = (p & t).sum().float()
        fp = (p & ~t).sum().float()
        fn = (~p & t).sum().float()
        tn = (~p & ~t).sum().float()

        dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        sen = tp / (tp + fn + 1e-6)
        spe = tn / (tn + fp + 1e-6)

        metrics[f"dice_c{c}"] = dice.item()
        metrics[f"iou_c{c}"] = iou.item()
        metrics[f"sen_c{c}"] = sen.item()
        metrics[f"spe_c{c}"] = spe.item()

        dice_vals.append(dice)
        iou_vals.append(iou)
        sen_vals.append(sen)
        spe_vals.append(spe)

    metrics["dice_mean"] = torch.stack(dice_vals).mean().item()
    metrics["iou_mean"] = torch.stack(iou_vals).mean().item()
    metrics["sen_mean"] = torch.stack(sen_vals).mean().item()
    metrics["spe_mean"] = torch.stack(spe_vals).mean().item()
    return metrics
