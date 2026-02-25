from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Dict

import nibabel as nib
import torch
from torch.utils.data import DataLoader

from brats_seg.data.dataset import BraTSDataset, PreprocessConfig
from brats_seg.data.index import CaseRecord
from brats_seg.models.factory import build_model
from brats_seg.training.artifacts import (
    save_feature_attention_artifacts,
    save_prediction_nifti,
    save_val_sample,
    write_artifact_manifest,
    write_case_metrics,
    write_volume_stats,
    write_worst_cases,
)
from brats_seg.training.engine import validate
from brats_seg.training.losses import DiceCELoss, DiceLoss
from brats_seg.utils import append_jsonl, load_json, save_json, setup_logger


def _records_to_cases(records: list[dict]) -> list[CaseRecord]:
    return [
        CaseRecord(
            case_id=r["case_id"],
            grade=r["grade"],
            image_paths=r["image_paths"],
            label_path=r["label_path"],
        )
        for r in records
    ]


def _center_crop_5d(x: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    _, _, d, h, w = x.shape
    pd, ph, pw = patch_size
    z0 = max((d - pd) // 2, 0)
    y0 = max((h - ph) // 2, 0)
    x0 = max((w - pw) // 2, 0)
    z1 = min(z0 + pd, d)
    y1 = min(y0 + ph, h)
    x1 = min(x0 + pw, w)
    return x[:, :, z0:z1, y0:y1, x0:x1]


def _build_case_meta(cases: list[CaseRecord]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for c in cases:
        hdr = nib.load(c.label_path).header
        zooms = hdr.get_zooms()
        voxel_volume_mm3 = float(zooms[0] * zooms[1] * zooms[2]) if len(zooms) >= 3 else 1.0
        out[c.case_id] = {
            "case_id": c.case_id,
            "grade": c.grade,
            "label_path": c.label_path,
            "voxel_volume_mm3": voxel_volume_mm3,
        }
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split-json", type=str, default="artifacts/splits/split_70_20_10.json")
    p.add_argument("--model", type=str, default="improved_unet3d", choices=["improved_unet3d", "vnet", "unetr"])
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--disable-pin-memory", action="store_true")
    p.add_argument("--out-dir", type=str, default="artifacts/eval")
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-val-samples", type=int, default=2)
    p.add_argument("--save-feature-attn-samples", type=int, default=1)
    p.add_argument("--save-pred-nifti-max-cases", type=int, default=999999)
    p.add_argument("--save-worst-cases", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("eval", str(out / "eval.log"), level=args.log_level)
    profile_path = out / "profile_eval.jsonl"

    patch_size = tuple(args.patch_size)
    if args.model == "vnet" and patch_size != (128, 128, 128):
        patch_size = (128, 128, 128)
        logger.info("VNet patch size is fixed to 128x128x128 (report-aligned).")
    if args.model == "unetr" and patch_size == (128, 128, 128):
        patch_size = (128, 128, 256)
        logger.info("UNETR patch size auto-adjusted to 128x128x256 for sequence length 1024.")

    payload = load_json(args.split_json)
    test_cases = _records_to_cases(payload["records"]["test"])
    test_case_meta = _build_case_meta(test_cases)

    pin_memory = not args.disable_pin_memory
    ds = BraTSDataset(
        test_cases,
        patch_size=patch_size,
        train=False,
        preprocess_cfg=PreprocessConfig(),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device selected: %s", device)
    if device.type == "cuda":
        logger.info("CUDA device: %s", torch.cuda.get_device_name(device))

    model = build_model(args.model, num_classes=4).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    criterion = DiceLoss(num_classes=4) if args.model == "vnet" else DiceCELoss(num_classes=4)

    artifact_records: list[dict] = []
    sample_count = 0
    feature_attn_count = 0
    nifti_count = 0
    callback_budget = max(
        args.save_val_samples,
        args.save_feature_attn_samples,
        args.save_pred_nifti_max_cases,
    )

    def _profile_callback(payload: dict):
        append_jsonl(str(profile_path), payload)

    def _eval_callback(batch_idx, case_id, image_cpu, label_cpu, logits_cpu, case_metric_row):
        nonlocal sample_count, feature_attn_count, nifti_count
        image_np = image_cpu[0].numpy()
        label_np = label_cpu[0].numpy()
        logits_np = logits_cpu[0].numpy()
        pred_np = logits_np.argmax(axis=0)
        record = {"case_id": case_id, "sample_idx": int(batch_idx)}

        if sample_count < args.save_val_samples:
            sample_files = save_val_sample(
                out_dir=str(out),
                epoch=0,
                sample_idx=batch_idx,
                case_id=case_id,
                image=image_np,
                label=label_np,
                pred=pred_np,
                logits=logits_np,
            )
            sample_count += 1
            record.update(sample_files)

        meta = test_case_meta.get(case_id, {})
        if nifti_count < args.save_pred_nifti_max_cases and meta.get("label_path"):
            nifti_path = save_prediction_nifti(
                pred=pred_np,
                out_dir=str(out),
                epoch=0,
                case_id=case_id,
                ref_nifti_path=str(meta["label_path"]),
            )
            nifti_count += 1
            record["pred_nifti"] = nifti_path

        if feature_attn_count < args.save_feature_attn_samples and hasattr(model, "forward_with_intermediates"):
            with torch.no_grad():
                center_patch = _center_crop_5d(image_cpu, patch_size=patch_size).to(device, non_blocking=True)
                _, intermediates = model.forward_with_intermediates(center_patch)
                feature_maps = {
                    k: v.detach().float().mean(dim=1).cpu().numpy()[0] for k, v in intermediates.get("features", {}).items()
                }
                attention_maps = {
                    k: v.detach().float().cpu().numpy()[0, 0] for k, v in intermediates.get("attentions", {}).items()
                }
            fa_files = save_feature_attention_artifacts(
                out_dir=str(out),
                epoch=0,
                sample_idx=batch_idx,
                case_id=case_id,
                feature_maps=feature_maps,
                attention_maps=attention_maps,
            )
            feature_attn_count += 1
            record.update(fa_files)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if len(record) > 2:
            artifact_records.append(record)

    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    metrics = validate(
        model,
        loader,
        criterion,
        device,
        patch_size=patch_size,
        use_sliding_window=True,
        overlap=0.5,
        batch_callback=_eval_callback if callback_budget > 0 else None,
        max_callback_batches=callback_budget,
        case_ids=[c.case_id for c in test_cases],
        profile_callback=_profile_callback,
        logger=logger,
        log_interval=args.log_interval,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    case_details = metrics.pop("case_details", [])
    artifact_by_case = {r["case_id"]: r for r in artifact_records}
    case_metric_rows = []
    volume_rows = []
    for row in case_details:
        case_id = str(row.get("case_id", ""))
        meta = test_case_meta.get(case_id, {})
        voxel_volume_mm3 = float(meta.get("voxel_volume_mm3", 1.0))
        enriched = dict(row)
        enriched["grade"] = meta.get("grade", "")
        enriched["voxel_volume_mm3"] = voxel_volume_mm3

        for cls in (1, 2, 3):
            gt_key = f"gt_c{cls}_voxels"
            pr_key = f"pred_c{cls}_voxels"
            df_key = f"diff_c{cls}_voxels"
            gt_vox = float(enriched.get(gt_key, 0.0))
            pr_vox = float(enriched.get(pr_key, 0.0))
            df_vox = float(enriched.get(df_key, pr_vox - gt_vox))
            enriched[f"gt_c{cls}_mm3"] = gt_vox * voxel_volume_mm3
            enriched[f"pred_c{cls}_mm3"] = pr_vox * voxel_volume_mm3
            enriched[f"diff_c{cls}_mm3"] = df_vox * voxel_volume_mm3

        gt_tumor_vox = float(enriched.get("gt_tumor_voxels", 0.0))
        pr_tumor_vox = float(enriched.get("pred_tumor_voxels", 0.0))
        df_tumor_vox = float(enriched.get("diff_tumor_voxels", pr_tumor_vox - gt_tumor_vox))
        enriched["gt_tumor_mm3"] = gt_tumor_vox * voxel_volume_mm3
        enriched["pred_tumor_mm3"] = pr_tumor_vox * voxel_volume_mm3
        enriched["diff_tumor_mm3"] = df_tumor_vox * voxel_volume_mm3

        if case_id in artifact_by_case:
            enriched.update(artifact_by_case[case_id])
        case_metric_rows.append(enriched)
        volume_rows.append(
            {
                "case_id": case_id,
                "grade": enriched.get("grade", ""),
                "voxel_volume_mm3": voxel_volume_mm3,
                "gt_c1_voxels": enriched.get("gt_c1_voxels", 0),
                "pred_c1_voxels": enriched.get("pred_c1_voxels", 0),
                "diff_c1_voxels": enriched.get("diff_c1_voxels", 0),
                "gt_c2_voxels": enriched.get("gt_c2_voxels", 0),
                "pred_c2_voxels": enriched.get("pred_c2_voxels", 0),
                "diff_c2_voxels": enriched.get("diff_c2_voxels", 0),
                "gt_c3_voxels": enriched.get("gt_c3_voxels", 0),
                "pred_c3_voxels": enriched.get("pred_c3_voxels", 0),
                "diff_c3_voxels": enriched.get("diff_c3_voxels", 0),
                "gt_tumor_voxels": enriched.get("gt_tumor_voxels", 0),
                "pred_tumor_voxels": enriched.get("pred_tumor_voxels", 0),
                "diff_tumor_voxels": enriched.get("diff_tumor_voxels", 0),
                "gt_c1_mm3": enriched.get("gt_c1_mm3", 0.0),
                "pred_c1_mm3": enriched.get("pred_c1_mm3", 0.0),
                "diff_c1_mm3": enriched.get("diff_c1_mm3", 0.0),
                "gt_c2_mm3": enriched.get("gt_c2_mm3", 0.0),
                "pred_c2_mm3": enriched.get("pred_c2_mm3", 0.0),
                "diff_c2_mm3": enriched.get("diff_c2_mm3", 0.0),
                "gt_c3_mm3": enriched.get("gt_c3_mm3", 0.0),
                "pred_c3_mm3": enriched.get("pred_c3_mm3", 0.0),
                "diff_c3_mm3": enriched.get("diff_c3_mm3", 0.0),
                "gt_tumor_mm3": enriched.get("gt_tumor_mm3", 0.0),
                "pred_tumor_mm3": enriched.get("pred_tumor_mm3", 0.0),
                "diff_tumor_mm3": enriched.get("diff_tumor_mm3", 0.0),
            }
        )

    case_metrics_files = write_case_metrics(str(out), epoch=0, rows=case_metric_rows)
    volume_files = write_volume_stats(str(out), epoch=0, rows=volume_rows)
    worst_files = write_worst_cases(
        out_dir=str(out),
        epoch=0,
        rows=case_metric_rows,
        top_k=args.save_worst_cases,
        score_key="dice_mean",
    )
    manifest_path = write_artifact_manifest(str(out), epoch=0, records=artifact_records) if artifact_records else ""

    save_json(
        str(out / "metrics.json"),
        {
            "metrics": metrics,
            "model": args.model,
            "ckpt": args.ckpt,
            "case_metrics": case_metrics_files,
            "volume_stats": volume_files,
            "worst_cases": worst_files,
            "artifact_manifest": manifest_path,
            "profile_jsonl": str(profile_path),
        },
    )
    logger.info("Evaluation done: %s", metrics)
    logger.info("Case metrics csv: %s", case_metrics_files.get("case_metrics_csv", ""))
    logger.info("Volume stats csv: %s", volume_files.get("volume_stats_csv", ""))
    logger.info("Worst cases csv: %s", worst_files.get("worst_cases_csv", ""))
    if manifest_path:
        logger.info("Artifact manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
