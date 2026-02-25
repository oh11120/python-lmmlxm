from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=str, required=True)
    p.add_argument("--engine", type=str, default="artifacts/trt/improved_unet3d_fp16.engine")
    p.add_argument("--workspace", type=int, default=4096, help="workspace size in MB")
    args = p.parse_args()

    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise RuntimeError("`trtexec` not found. Please install TensorRT and add trtexec to PATH.")

    onnx = Path(args.onnx)
    if not onnx.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx}")

    engine = Path(args.engine)
    engine.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        trtexec,
        f"--onnx={onnx}",
        f"--saveEngine={engine}",
        "--fp16",
        f"--workspace={args.workspace}",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Saved TensorRT engine: {engine}")


if __name__ == "__main__":
    main()
