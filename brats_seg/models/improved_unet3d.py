from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import AttentionGate, DoubleConv, UpBlock


class ImprovedUNet3D(nn.Module):
    """3D U-Net with level-2 modality fusion and attention-gated skip connections."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.pool = nn.MaxPool3d(2)

        # Stage-1: modality-specific 1->64
        self.enc1_mod = nn.ModuleList([DoubleConv(1, 64) for _ in range(4)])
        self.skip1_fuse = nn.Sequential(
            nn.Conv3d(64 * 4, 64, kernel_size=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # Stage-2: modality-specific 64->128 then fusion 512->256
        self.enc2_mod = nn.ModuleList([DoubleConv(64, 128) for _ in range(4)])
        self.modality_fusion = nn.Sequential(
            nn.Conv3d(128 * 4, 256, kernel_size=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

        # Unified encoder: 256 -> 256 -> 512 -> 1024
        self.enc3 = DoubleConv(256, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = UpBlock(1024, 512)
        self.ag4 = AttentionGate(skip_ch=512, gate_ch=512, inter_ch=16)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = UpBlock(512, 256)
        self.ag3 = AttentionGate(skip_ch=256, gate_ch=256, inter_ch=16)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = UpBlock(256, 256)
        self.skip2_proj = nn.Conv3d(256, 256, kernel_size=1, bias=False)
        self.ag2 = AttentionGate(skip_ch=256, gate_ch=256, inter_ch=16)
        self.dec2 = DoubleConv(512, 256)

        self.up1 = UpBlock(256, 256)
        self.skip1_proj = nn.Conv3d(64, 256, kernel_size=1, bias=False)
        self.ag1 = AttentionGate(skip_ch=256, gate_ch=256, inter_ch=16)
        self.dec1 = DoubleConv(512, 256)

        # Report: final 1x1x1 maps 256-dim features to 4 classes.
        self.out = nn.Conv3d(256, num_classes, kernel_size=1)

    def forward_with_intermediates(self, x: torch.Tensor):
        # x: [B,4,D,H,W]
        x1_mod = [self.enc1_mod[i](x[:, i : i + 1]) for i in range(4)]
        x1 = self.skip1_fuse(torch.cat(x1_mod, dim=1))

        x2_mod = [self.enc2_mod[i](self.pool(x1_mod[i])) for i in range(4)]
        x2 = self.modality_fusion(torch.cat(x2_mod, dim=1))

        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        xb = self.bottleneck(self.pool(x4))

        d4 = self.up4(xb)
        s4, a4 = self.ag4(x4, d4, return_mask=True)
        d4 = self.dec4(torch.cat([d4, s4], dim=1))

        d3 = self.up3(d4)
        s3, a3 = self.ag3(x3, d3, return_mask=True)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))

        d2 = self.up2(d3)
        s2, a2 = self.ag2(self.skip2_proj(x2), d2, return_mask=True)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))

        d1 = self.up1(d2)
        s1, a1 = self.ag1(self.skip1_proj(x1), d1, return_mask=True)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))

        logits = self.out(d1)
        intermediates = {
            "features": {
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "x4": x4,
                "xb": xb,
                "d3": d3,
                "d2": d2,
                "d1": d1,
            },
            "attentions": {
                "ag4": a4,
                "ag3": a3,
                "ag2": a2,
                "ag1": a1,
            },
        }
        return logits, intermediates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_intermediates(x)
        return logits
