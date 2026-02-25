from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from .infer import sliding_window_inference
from .metrics import compute_segmentation_metrics


@dataclass
class TrainState:
    best_dice: float = -1.0
    patience_count: int = 0


def _cuda_mem_summary(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return ""
    allocated_gb = torch.cuda.memory_allocated(device=device) / (1024**3)
    reserved_gb = torch.cuda.memory_reserved(device=device) / (1024**3)
    return f" cuda_alloc={allocated_gb:.2f}G cuda_reserved={reserved_gb:.2f}G"


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
    log_interval: int = 0,
    profile_callback: Optional[Callable[[Dict[str, float]], None]] = None,
) -> float:
    model.train()
    total = 0.0
    count = 0
    num_batches = len(loader)
    epoch_start = time.perf_counter()
    end = epoch_start
    for batch_idx, (image, label) in enumerate(loader, start=1):
        data_time = time.perf_counter() - end
        batch_start = time.perf_counter()
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total += loss.item()
        count += 1
        if logger is not None and log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == num_batches):
            logger.info(
                "train_step=%d/%d loss=%.4f avg_loss=%.4f data_sec=%.2f iter_sec=%.2f%s",
                batch_idx,
                num_batches,
                loss.item(),
                total / max(1, count),
                data_time,
                time.perf_counter() - batch_start,
                _cuda_mem_summary(device),
            )
        if profile_callback is not None:
            row = {
                "phase": "train",
                "batch": batch_idx,
                "num_batches": num_batches,
                "loss": float(loss.item()),
                "avg_loss": float(total / max(1, count)),
                "data_time_sec": float(data_time),
                "iter_time_sec": float(time.perf_counter() - batch_start),
                "cuda_alloc_gb": float(torch.cuda.memory_allocated(device=device) / (1024**3))
                if device.type == "cuda" and torch.cuda.is_available()
                else 0.0,
                "cuda_reserved_gb": float(torch.cuda.memory_reserved(device=device) / (1024**3))
                if device.type == "cuda" and torch.cuda.is_available()
                else 0.0,
            }
            profile_callback(row)
        end = time.perf_counter()
    if logger is not None:
        logger.info(
            "train_epoch_done batches=%d avg_loss=%.4f epoch_sec=%.2f%s",
            num_batches,
            total / max(1, count),
            time.perf_counter() - epoch_start,
            _cuda_mem_summary(device),
        )
    return total / max(1, count)


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    use_sliding_window: bool = True,
    overlap: float = 0.5,
    batch_callback: Optional[Callable[[int, str, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]], None]] = None,
    max_callback_batches: int = 0,
    case_ids: Optional[Sequence[str]] = None,
    profile_callback: Optional[Callable[[Dict[str, float]], None]] = None,
    logger: Optional[logging.Logger] = None,
    log_interval: int = 0,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    agg = None
    per_case = []

    with torch.no_grad():
        callback_count = 0
        num_batches = len(loader)
        val_start = time.perf_counter()
        end = val_start
        for batch_idx, (image, label) in enumerate(loader):
            step_no = batch_idx + 1
            case_id = case_ids[batch_idx] if case_ids is not None and batch_idx < len(case_ids) else f"case_{batch_idx:04d}"
            data_time = time.perf_counter() - end
            step_start = time.perf_counter()
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            if use_sliding_window:
                logits = sliding_window_inference(model, image, patch_size=patch_size, overlap=overlap)
            else:
                logits = model(image)
            loss = criterion(logits, label)
            m = compute_segmentation_metrics(logits, label)
            pred = torch.argmax(logits, dim=1)
            row = {"case_id": case_id, "val_loss": float(loss.item()), **{k: float(v) for k, v in m.items()}}
            for cls in (1, 2, 3):
                row[f"gt_c{cls}_voxels"] = int((label == cls).sum().item())
                row[f"pred_c{cls}_voxels"] = int((pred == cls).sum().item())
                row[f"diff_c{cls}_voxels"] = row[f"pred_c{cls}_voxels"] - row[f"gt_c{cls}_voxels"]
            row["gt_tumor_voxels"] = int((label > 0).sum().item())
            row["pred_tumor_voxels"] = int((pred > 0).sum().item())
            row["diff_tumor_voxels"] = row["pred_tumor_voxels"] - row["gt_tumor_voxels"]
            per_case.append(row)

            if batch_callback is not None and callback_count < max_callback_batches:
                batch_callback(
                    batch_idx,
                    case_id,
                    image.detach().cpu(),
                    label.detach().cpu(),
                    logits.detach().cpu(),
                    row,
                )
                callback_count += 1

            total_loss += loss.item()
            count += 1
            if agg is None:
                agg = {k: 0.0 for k in m}
            for k, v in m.items():
                agg[k] += v

            if logger is not None and log_interval > 0 and (step_no % log_interval == 0 or step_no == num_batches):
                dice_running = (agg.get("dice_mean", 0.0) / max(1, count)) if agg is not None else 0.0
                iou_running = (agg.get("iou_mean", 0.0) / max(1, count)) if agg is not None else 0.0
                logger.info(
                    "val_step=%d/%d val_loss=%.4f dice=%.4f iou=%.4f step_sec=%.2f%s",
                    step_no,
                    num_batches,
                    total_loss / max(1, count),
                    dice_running,
                    iou_running,
                    time.perf_counter() - step_start,
                    _cuda_mem_summary(device),
                )
            if profile_callback is not None:
                profile_callback(
                    {
                        "phase": "val",
                        "batch": step_no,
                        "num_batches": num_batches,
                        "case_id": case_id,
                        "val_loss": float(total_loss / max(1, count)),
                        "dice_running": float((agg.get("dice_mean", 0.0) / max(1, count)) if agg is not None else 0.0),
                        "iou_running": float((agg.get("iou_mean", 0.0) / max(1, count)) if agg is not None else 0.0),
                        "data_time_sec": float(data_time),
                        "iter_time_sec": float(time.perf_counter() - step_start),
                        "cuda_alloc_gb": float(torch.cuda.memory_allocated(device=device) / (1024**3))
                        if device.type == "cuda" and torch.cuda.is_available()
                        else 0.0,
                        "cuda_reserved_gb": float(torch.cuda.memory_reserved(device=device) / (1024**3))
                        if device.type == "cuda" and torch.cuda.is_available()
                        else 0.0,
                    }
                )
            end = time.perf_counter()

    out = {"val_loss": total_loss / max(1, count)}
    if agg is not None:
        out.update({k: v / max(1, count) for k, v in agg.items()})
    out["case_details"] = per_case
    if logger is not None:
        logger.info(
            "val_epoch_done batches=%d val_loss=%.4f epoch_sec=%.2f%s",
            num_batches,
            out["val_loss"],
            time.perf_counter() - val_start,
            _cuda_mem_summary(device),
        )
    return out


def maybe_save_best(
    model: torch.nn.Module,
    metrics: Dict[str, float],
    state: TrainState,
    output_dir: str,
    early_stopping_patience: int,
) -> bool:
    dice = metrics.get("dice_mean", -1.0)
    improved = dice > state.best_dice
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if improved:
        state.best_dice = dice
        state.patience_count = 0
        torch.save(model.state_dict(), out / "best_model.pt")
    else:
        state.patience_count += 1

    return state.patience_count >= early_stopping_patience
