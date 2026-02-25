from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brats_seg.models.factory import build_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="improved_unet3d", choices=["improved_unet3d", "vnet", "unetr"])
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--output", type=str, default="artifacts/onnx/improved_unet3d.onnx")
    p.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    args = p.parse_args()

    model = build_model(args.model, num_classes=4)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    d, h, w = args.patch_size
    dummy = torch.randn(1, 4, d, h, w)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={
            "input": {0: "batch", 2: "depth", 3: "height", 4: "width"},
            "logits": {0: "batch", 2: "depth", 3: "height", 4: "width"},
        },
    )
    print(f"Exported ONNX to {out}")


if __name__ == "__main__":
    main()
