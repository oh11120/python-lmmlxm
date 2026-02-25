from __future__ import annotations

import torch.nn as nn

from .improved_unet3d import ImprovedUNet3D
from .reported_vnet import ReportVNet


def build_model(name: str, num_classes: int = 4) -> nn.Module:
    name = name.lower()
    if name in {"improved_unet3d", "unet", "main"}:
        return ImprovedUNet3D(num_classes=num_classes)

    if name in {"vnet", "unetr"}:
        if name == "vnet":
            # Report-aligned 3D V-Net channel schedule: 16->32->64->128->256.
            return ReportVNet(in_channels=4, out_channels=num_classes)

        try:
            from monai.networks.nets import UNETR
        except Exception as exc:
            raise ImportError(
                "UNETR requires MONAI. Install with `pip install monai` and retry."
            ) from exc

        # Report-aligned core Transformer settings: 12 layers, 12 heads, embed=768.
        kwargs = dict(
            in_channels=4,
            out_channels=num_classes,
            # 8 x 8 x 16 tokens with patch size 16 -> sequence length 1024.
            img_size=(128, 128, 256),
            feature_size=64,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            norm_name="instance",
            spatial_dims=3,
        )
        try:
            return UNETR(**kwargs, proj_type="perceptron")
        except TypeError:
            return UNETR(**kwargs, pos_embed="perceptron")

    raise ValueError(f"Unsupported model: {name}")
