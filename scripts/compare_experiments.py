from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def records_to_cases(records: List[dict]):
    from brats_seg.data.index import CaseRecord

    return [
        CaseRecord(
            case_id=r["case_id"],
            grade=r["grade"],
            image_paths=r["image_paths"],
            label_path=r["label_path"],
        )
        for r in records
    ]


def parse_experiment(s: str) -> Tuple[str, str, str]:
    # format: name:model:ckpt
    parts = s.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid --exp format: {s}. expected name:model:ckpt")
    return parts[0], parts[1], parts[2]


def evaluate_one(
    model_name: str,
    ckpt: str,
    test_cases,
    patch_size: Tuple[int, int, int],
    num_workers: int,
) -> Dict[str, float]:
    import torch
    from torch.utils.data import DataLoader
    from brats_seg.data.dataset import BraTSDataset, PreprocessConfig
    from brats_seg.models.factory import build_model
    from brats_seg.training.engine import validate
    from brats_seg.training.losses import DiceCELoss, DiceLoss

    if model_name.lower() == "vnet" and patch_size != (128, 128, 128):
        patch_size = (128, 128, 128)
    if model_name.lower() == "unetr" and patch_size == (128, 128, 128):
        patch_size = (128, 128, 256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_name, num_classes=4).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    ds = BraTSDataset(
        test_cases,
        patch_size=patch_size,
        train=False,
        preprocess_cfg=PreprocessConfig(),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    criterion = DiceLoss(num_classes=4) if model_name.lower() == "vnet" else DiceCELoss(num_classes=4)
    metrics = validate(
        model,
        loader,
        criterion,
        device,
        patch_size=patch_size,
        use_sliding_window=True,
        overlap=0.5,
    )
    return metrics


def write_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "model",
        "ckpt",
        "val_loss",
        "dice_mean",
        "iou_mean",
        "sen_mean",
        "spe_mean",
        "dice_c1",
        "dice_c2",
        "dice_c3",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_markdown(rows: List[Dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Experiment | Model | Dice(mean) | IoU(mean) | SEN(mean) | SPE(mean) | Dice(c1) | Dice(c2) | Dice(c3) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {model} | {dice_mean:.4f} | {iou_mean:.4f} | {sen_mean:.4f} | {spe_mean:.4f} | {dice_c1:.4f} | {dice_c2:.4f} | {dice_c3:.4f} |".format(
                experiment=row["experiment"],
                model=row["model"],
                dice_mean=float(row.get("dice_mean", 0.0)),
                iou_mean=float(row.get("iou_mean", 0.0)),
                sen_mean=float(row.get("sen_mean", 0.0)),
                spe_mean=float(row.get("spe_mean", 0.0)),
                dice_c1=float(row.get("dice_c1", 0.0)),
                dice_c2=float(row.get("dice_c2", 0.0)),
                dice_c3=float(row.get("dice_c3", 0.0)),
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_experiments() -> List[Tuple[str, str, str]]:
    return [
        ("main", "improved_unet3d", "artifacts/exp_main/best_model.pt"),
        ("vnet", "vnet", "artifacts/exp_vnet/best_model.pt"),
        ("unetr", "unetr", "artifacts/exp_unetr/best_model.pt"),
    ]


def main() -> None:
    from brats_seg.utils import load_json, save_json

    p = argparse.ArgumentParser()
    p.add_argument("--split-json", type=str, default="artifacts/splits/split_70_20_10.json")
    p.add_argument("--exp", action="append", default=[], help="name:model:ckpt. can repeat")
    p.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-dir", type=str, default="artifacts/reports")
    args = p.parse_args()

    payload = load_json(args.split_json)
    test_cases = records_to_cases(payload["records"]["test"])
    patch_size = tuple(args.patch_size)

    experiments = [parse_experiment(x) for x in args.exp] if args.exp else default_experiments()

    rows: List[Dict[str, object]] = []
    for exp_name, model_name, ckpt in experiments:
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            print(f"[skip] {exp_name}: checkpoint not found -> {ckpt}")
            continue

        print(f"[run] {exp_name}: model={model_name}, ckpt={ckpt}")
        try:
            metrics = evaluate_one(
                model_name=model_name,
                ckpt=str(ckpt_path),
                test_cases=test_cases,
                patch_size=patch_size,
                num_workers=args.num_workers,
            )
        except Exception as exc:
            print(f"[fail] {exp_name}: {type(exc).__name__}: {exc}")
            continue

        row: Dict[str, object] = {
            "experiment": exp_name,
            "model": model_name,
            "ckpt": str(ckpt_path),
        }
        row.update(metrics)
        rows.append(row)
        print(f"[ok] {exp_name}: dice_mean={metrics.get('dice_mean', 0.0):.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(str(out_dir / "comparison.json"), {"rows": rows})
    write_csv(rows, out_dir / "comparison.csv")
    write_markdown(rows, out_dir / "comparison.md")

    print(f"Saved: {out_dir / 'comparison.json'}")
    print(f"Saved: {out_dir / 'comparison.csv'}")
    print(f"Saved: {out_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
