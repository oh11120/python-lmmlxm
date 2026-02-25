from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class AttentionGate(nn.Module):
    def __init__(self, skip_ch: int, gate_ch: int, inter_ch: int = 16):
        super().__init__()
        if skip_ch != gate_ch:
            raise ValueError("Report-style attention gate requires skip_ch == gate_ch")
        self.mul_conv = nn.Sequential(
            nn.Conv3d(skip_ch, inter_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(inter_ch),
            nn.ReLU(inplace=True),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_ch, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x_skip: torch.Tensor, g: torch.Tensor, return_mask: bool = False):
        if x_skip.shape != g.shape:
            raise ValueError(
                f"Attention inputs must share shape for element-wise multiplication, got {tuple(x_skip.shape)} vs {tuple(g.shape)}"
            )
        attn = self.mul_conv(x_skip * g)
        mask = self.psi(attn)
        out = x_skip * mask
        if return_mask:
            return out, mask
        return out
