from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(out_ch),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.PReLU(out_ch)
        self.conv = ConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.down(x)))
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.PReLU(out_ch)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.up(x)))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ReportVNet(nn.Module):
    """V-Net variant aligned to opening report channels: 16->32->64->128->256."""

    def __init__(self, in_channels: int = 4, out_channels: int = 4):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 16)
        self.enc2 = DownBlock(16, 32)
        self.enc3 = DownBlock(32, 64)
        self.enc4 = DownBlock(64, 128)
        self.enc5 = DownBlock(128, 256)

        self.dec4 = UpBlock(256, 128, 128)
        self.dec3 = UpBlock(128, 64, 64)
        self.dec2 = UpBlock(64, 32, 32)
        self.dec1 = UpBlock(32, 16, 16)

        self.out = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.enc5(e4)

        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.out(d1)
