from __future__ import annotations

import argparse
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict

import nibabel as nib
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from brats_seg.data.dataset import BraTSDataset, PreprocessConfig
from brats_seg.data.index import CaseRecord, discover_cases
from brats_seg.data.split import filter_cases
from brats_seg.models.factory import build_model
from brats_seg.training.engine import TrainState, maybe_save_best, run_epoch, validate
from brats_seg.training.losses import DiceCELoss, DiceLoss
from brats_seg.training.artifacts import (
    save_feature_attention_artifacts,
    save_prediction_nifti,
    save_val_sample,
    write_artifact_manifest,
    write_case_metrics,
    write_volume_stats,
    write_worst_cases,
)
from brats_seg.utils import append_jsonl, load_json, save_csv, save_json, set_seed, setup_logger


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
    p.add_argument("--data-root", type=str, default="data/MICCAI_BraTS_2019_Data_Training")
    p.add_argument("--split-json", type=str, default="artifacts/splits/split_70_20_10.json")
    p.add_argument("--out-dir", type=str, default="artifacts/exp_main")
    p.add_argument("--model", type=str, default="improved_unet3d", choices=["improved_unet3d", "vnet", "unetr"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--early-stop", type=int, default=5)
    p.add_argument("--use-n4", action="store_true")
    p.add_argument("--disable-n4", action="store_true")
    p.add_argument("--disable-denoise", action="store_true")
    p.add_argument("--disable-pin-memory", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-val-samples", type=int, default=1)
    p.add_argument("--save-val-every", type=int, default=1)
    p.add_argument("--save-feature-attn-samples", type=int, default=1)
    p.add_argument("--save-val-nifti-max-cases", type=int, default=2)
    p.add_argument("--save-worst-cases", type=int, default=10)
    p.add_argument("--save-checkpoint-every", type=int, default=1)
    p.add_argument("--disable-epoch-checkpoints", action="store_true")
    p.add_argument("--resume", type=str, default="")
    return p.parse_args()


def prepare_cases(args) -> tuple[list[CaseRecord], list[CaseRecord], list[CaseRecord]]:
    split_path = Path(args.split_json)
    if split_path.exists():
        payload = load_json(str(split_path))
        train_cases = _records_to_cases(payload["records"]["train"])
        val_cases = _records_to_cases(payload["records"]["val"])
        test_cases = _records_to_cases(payload["records"]["test"])
        return train_cases, val_cases, test_cases

    from brats_seg.data.split import stratified_split

    cases = discover_cases(args.data_root)
    split = stratified_split(cases, seed=args.seed)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(
        str(split_path),
        {
            "meta": {"seed": args.seed},
            "records": {
                "train": [c.__dict__ for c in filter_cases(cases, split["train"])],
                "val": [c.__dict__ for c in filter_cases(cases, split["val"])],
                "test": [c.__dict__ for c in filter_cases(cases, split["test"])],
            },
        },
    )
    return (
        filter_cases(cases, split["train"]),
        filter_cases(cases, split["val"]),
        filter_cases(cases, split["test"]),
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_cases, val_cases, _ = prepare_cases(args)
    val_case_meta = _build_case_meta(val_cases)
    patch_size = tuple(args.patch_size)
    if args.model == "vnet" and patch_size != (128, 128, 128):
        patch_size = (128, 128, 128)
        print("VNet patch size is fixed to 128x128x128 (report-aligned).")
    if args.model == "unetr" and patch_size == (128, 128, 128):
        patch_size = (128, 128, 256)
        print("UNETR patch size auto-adjusted to 128x128x256 for sequence length 1024.")
    use_n4 = True
    if args.disable_n4:
        use_n4 = False
    if args.use_n4:
        use_n4 = True
    preprocess_cfg = PreprocessConfig(use_denoise=not args.disable_denoise, use_n4=use_n4)
    pin_memory = not args.disable_pin_memory

    train_ds = BraTSDataset(
        train_cases,
        patch_size=patch_size,
        train=True,
        preprocess_cfg=preprocess_cfg,
        base_seed=args.seed,
    )
    val_ds = BraTSDataset(
        val_cases,
        patch_size=patch_size,
        train=False,
        preprocess_cfg=preprocess_cfg,
        base_seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("train", str(out / "logs" / "train.log"), level=args.log_level)
    metrics_jsonl_path = out / "logs" / "metrics.jsonl"
    train_profile_path = out / "logs" / "profile_train.jsonl"
    val_profile_path = out / "logs" / "profile_val.jsonl"
    save_json(
        str(out / "run_config.json"),
        {
            "run_id": run_id,
            "args": vars(args),
            "patch_size": list(patch_size),
            "dataset_sizes": {"train": len(train_cases), "val": len(val_cases)},
            "model": args.model,
            "n4_enabled": use_n4,
        },
    )
    logger.info("Run initialized: run_id=%s out_dir=%s model=%s", run_id, args.out_dir, args.model)
    logger.info("Dataset size: train=%d val=%d patch_size=%s", len(train_cases), len(val_cases), patch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device selected: %s", device)
    if device.type == "cuda":
        logger.info("CUDA device: %s", torch.cuda.get_device_name(device))
    model = build_model(args.model, num_classes=4).to(device)
    if args.model == "vnet":
        criterion = DiceLoss(num_classes=4)
        logger.info("Using DiceLoss for VNet (report-aligned).")
    else:
        criterion = DiceCELoss(num_classes=4, ce_weight=0.5)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    history: list[dict] = []
    state = TrainState()
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(str(resume_path), map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        else:
            raise RuntimeError(f"Unsupported checkpoint format: {resume_path}")
        if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if isinstance(ckpt, dict) and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if isinstance(ckpt, dict):
            history = list(ckpt.get("history", []))
            state.best_dice = float(ckpt.get("best_dice", -1.0))
            state.patience_count = int(ckpt.get("patience_count", 0))
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            if "run_id" in ckpt:
                run_id = str(ckpt["run_id"])
        logger.info(
            "Resumed from %s at epoch=%d (next_epoch=%d) best_dice=%.4f history_len=%d",
            str(resume_path),
            start_epoch - 1,
            start_epoch,
            state.best_dice,
            len(history),
        )
        if start_epoch > args.epochs:
            logger.info("Checkpoint epoch already reached target epochs (%d). Nothing to train.", args.epochs)
            return
    logger.info(
        "Dataloader ready: train_batches=%d val_batches=%d batch_size=%d num_workers=%d train_pin_memory=%s val_pin_memory=%s log_interval=%d",
        len(train_loader),
        len(val_loader),
        args.batch_size,
        args.num_workers,
        pin_memory,
        False,
        args.log_interval,
    )
    logger.info(
        "Artifacts enabled: val_samples=%d feature_attn_samples=%d val_nifti_max_cases=%d worst_cases_topk=%d checkpoints=%s every=%d",
        args.save_val_samples,
        args.save_feature_attn_samples,
        args.save_val_nifti_max_cases,
        args.save_worst_cases,
        not args.disable_epoch_checkpoints,
        args.save_checkpoint_every,
    )
    if args.resume:
        save_json(
            str(out / "run_config.json"),
            {
                "run_id": run_id,
                "args": vars(args),
                "patch_size": list(patch_size),
                "dataset_sizes": {"train": len(train_cases), "val": len(val_cases)},
                "model": args.model,
                "n4_enabled": use_n4,
                "resume_from": args.resume,
                "resume_start_epoch": start_epoch,
            },
        )

    for epoch in range(start_epoch, args.epochs + 1):
        logger.info("Epoch %d/%d started", epoch, args.epochs)
        val_case_ids = [c.case_id for c in val_cases]

        def _train_profile_callback(payload: dict):
            payload["epoch"] = epoch
            append_jsonl(str(train_profile_path), payload)

        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            logger=logger,
            log_interval=args.log_interval,
            profile_callback=_train_profile_callback,
        )

        saved_samples: list[dict] = []
        artifact_records: list[dict] = []
        sample_count = 0
        feature_attn_count = 0
        nifti_count = 0
        should_export_val_artifacts = epoch % max(1, args.save_val_every) == 0
        callback_budget = 0
        if should_export_val_artifacts:
            callback_budget = max(
                args.save_val_samples,
                args.save_feature_attn_samples,
                args.save_val_nifti_max_cases,
            )

        def _val_profile_callback(payload: dict):
            payload["epoch"] = epoch
            append_jsonl(str(val_profile_path), payload)

        def _val_callback(batch_idx, case_id, image_cpu, label_cpu, logits_cpu, case_metric_row):
            nonlocal sample_count, feature_attn_count, nifti_count
            image_np = image_cpu[0].numpy()
            label_np = label_cpu[0].numpy()
            logits_np = logits_cpu[0].numpy()
            pred_np = logits_np.argmax(axis=0)
            record = {"case_id": case_id, "epoch": epoch, "sample_idx": int(batch_idx)}

            if sample_count < args.save_val_samples:
                sample_files = save_val_sample(
                    out_dir=str(out),
                    epoch=epoch,
                    sample_idx=batch_idx,
                    case_id=case_id,
                    image=image_np,
                    label=label_np,
                    pred=pred_np,
                    logits=logits_np,
                )
                sample_count += 1
                record.update(sample_files)
                saved_samples.append(sample_files)

            meta = val_case_meta.get(case_id, {})
            if nifti_count < args.save_val_nifti_max_cases and meta.get("label_path"):
                nifti_path = save_prediction_nifti(
                    pred=pred_np,
                    out_dir=str(out),
                    epoch=epoch,
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
                    epoch=epoch,
                    sample_idx=batch_idx,
                    case_id=case_id,
                    feature_maps=feature_maps,
                    attention_maps=attention_maps,
                )
                feature_attn_count += 1
                record.update(fa_files)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if len(record) > 3:
                artifact_records.append(record)

        if device.type == "cuda":
            # On low-VRAM GPUs, release cached blocks before full-volume validation.
            gc.collect()
            torch.cuda.empty_cache()
        val_metrics = validate(
            model,
            val_loader,
            criterion,
            device,
            patch_size=patch_size,
            use_sliding_window=True,
            overlap=0.5,
            batch_callback=_val_callback if callback_budget > 0 else None,
            max_callback_batches=callback_budget,
            case_ids=val_case_ids,
            profile_callback=_val_profile_callback,
            logger=logger,
            log_interval=args.log_interval,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        scheduler.step()

        case_details = val_metrics.pop("case_details", [])
        artifact_by_case = {}
        for rec in artifact_records:
            artifact_by_case[rec["case_id"]] = rec
        case_metric_rows = []
        volume_rows = []
        for row in case_details:
            case_id = str(row.get("case_id", ""))
            meta = val_case_meta.get(case_id, {})
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

        case_metrics_files = write_case_metrics(str(out), epoch, case_metric_rows)
        volume_files = write_volume_stats(str(out), epoch, volume_rows)
        worst_files = write_worst_cases(
            out_dir=str(out),
            epoch=epoch,
            rows=case_metric_rows,
            top_k=args.save_worst_cases,
            score_key="dice_mean",
        )
        logger.info(
            "Per-case artifacts: metrics=%s volume=%s worst=%s",
            case_metrics_files.get("case_metrics_csv", ""),
            volume_files.get("volume_stats_csv", ""),
            worst_files.get("worst_cases_csv", ""),
        )
        if artifact_records:
            manifest_path = write_artifact_manifest(str(out), epoch=epoch, records=artifact_records)
            logger.info("Artifact manifest saved: %s", manifest_path)

        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics, "lr": scheduler.get_last_lr()[0]}
        history.append(row)
        append_jsonl(str(metrics_jsonl_path), row)

        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f dice=%.4f iou=%.4f lr=%.6f",
            epoch,
            train_loss,
            val_metrics.get("val_loss", 0.0),
            val_metrics.get("dice_mean", 0.0),
            val_metrics.get("iou_mean", 0.0),
            scheduler.get_last_lr()[0],
        )
        if saved_samples:
            logger.info("Saved %d validation intermediate sample(s) at epoch=%d", len(saved_samples), epoch)

        if not args.disable_epoch_checkpoints and epoch % max(1, args.save_checkpoint_every) == 0:
            ckpt_dir = out / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            ckpt_payload = {
                "epoch": epoch,
                "run_id": run_id,
                "model": args.model,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": state.best_dice,
                "patience_count": state.patience_count,
                "history": history,
                "args": vars(args),
            }
            torch.save(ckpt_payload, ckpt_path)
            torch.save(ckpt_payload, ckpt_dir / "checkpoint_latest.pt")
            logger.info("Saved checkpoint snapshot: %s", ckpt_path)

        should_stop = maybe_save_best(
            model=model,
            metrics=val_metrics,
            state=state,
            output_dir=str(out),
            early_stopping_patience=args.early_stop,
        )
        save_json(str(out / "history.json"), {"history": history})
        save_csv(
            str(out / "history.csv"),
            history,
            fieldnames=["epoch", "train_loss", "val_loss", "dice_mean", "iou_mean", "sen_mean", "spe_mean", "lr"],
        )

        if should_stop:
            logger.info("Early stopping at epoch=%d", epoch)
            break

    torch.save(model.state_dict(), out / "last_model.pt")
    summary = {
        "run_id": run_id,
        "best_dice": state.best_dice,
        "epochs_ran": len(history),
        "output_dir": args.out_dir,
    }
    save_json(str(out / "summary.json"), summary)
    logger.info("Training done. best_dice=%.4f epochs=%d output=%s", state.best_dice, len(history), args.out_dir)


if __name__ == "__main__":
    main()
