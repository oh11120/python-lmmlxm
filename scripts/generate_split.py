from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brats_seg.data.index import discover_cases, to_serializable
from brats_seg.data.split import filter_cases, stratified_split
from brats_seg.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/MICCAI_BraTS_2019_Data_Training")
    parser.add_argument("--output", type=str, default="artifacts/splits/split_70_20_10.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = discover_cases(args.data_root)
    split = stratified_split(cases, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=args.seed)

    payload = {
        "meta": {
            "data_root": args.data_root,
            "seed": args.seed,
            "train_ratio": 0.7,
            "val_ratio": 0.2,
            "test_ratio": 0.1,
            "counts": {k: len(v) for k, v in split.items()},
        },
        "case_ids": split,
        "records": {
            k: to_serializable(filter_cases(cases, split[k])) for k in ("train", "val", "test")
        },
    }
    save_json(args.output, payload)
    print(f"Saved split: {args.output}")
    print(payload["meta"])


if __name__ == "__main__":
    main()
