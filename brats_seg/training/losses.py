from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    def __init__(self, num_classes: int = 4, smooth: float = 1e-5, ce_weight: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target)
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)
        inter = torch.sum(probs * one_hot, dims)
        den = torch.sum(probs + one_hot, dims)
        dice = (2 * inter + self.smooth) / (den + self.smooth)
        dice_loss = 1 - dice.mean()

        return self.ce_weight * ce + (1 - self.ce_weight) * dice_loss


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 4, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        inter = torch.sum(probs * one_hot, dims)
        den = torch.sum(probs + one_hot, dims)
        dice = (2 * inter + self.smooth) / (den + self.smooth)
        return 1 - dice.mean()
