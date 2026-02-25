from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import nibabel as nib
import numpy as np
from PIL import Image


@dataclass
class CaseArtifactPaths:
    case_id: str
    npz: str
    gt_png: str
    pred_png: str
    axial_png: str
    sagittal_png: str
    coronal_png: str
    entropy_png: str
    prob_c1_png: str
    prob_c2_png: str
    prob_c3_png: str
    feature_npz: Optional[str] = None
    feature_png: Optional[str] = None
    attention_npz: Optional[str] = None
    attention_png: Optional[str] = None
    pred_nifti: Optional[str] = None


def save_val_sample(
    out_dir: str,
    epoch: int,
    sample_idx: int,
    case_id: str,
    image: np.ndarray,
    label: np.ndarray,
    pred: np.ndarray,
    logits: np.ndarray,
) -> Dict[str, str]:
    """Save intermediate artifacts for paper writing/debugging.

    Inputs are single-sample arrays:
    - image: [C,D,H,W]
    - label/pred: [D,H,W]
    - logits: [K,D,H,W]
    """
    root = Path(out_dir)
    npz_dir = root / "intermediate" / "npz"
    fig_dir = root / "intermediate" / "figures"
    probs_dir = root / "intermediate" / "probability"
    entropy_dir = root / "intermediate" / "uncertainty"
    view_dir = root / "intermediate" / "views"
    npz_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    probs_dir.mkdir(parents=True, exist_ok=True)
    entropy_dir.mkdir(parents=True, exist_ok=True)
    view_dir.mkdir(parents=True, exist_ok=True)

    stem = f"epoch_{epoch:03d}_sample_{sample_idx:03d}_{case_id}"
    probs = _softmax_np(logits, axis=0)
    entropy = _entropy_np(probs, axis=0)
    max_prob = probs.max(axis=0)
    npz_path = npz_dir / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        image=image.astype(np.float32),
        label=label.astype(np.int16),
        pred=pred.astype(np.int16),
        logits=logits.astype(np.float32),
        probs=probs.astype(np.float32),
        entropy=entropy.astype(np.float32),
        max_prob=max_prob.astype(np.float32),
    )

    # Save representative middle slice overlay for quick visual inspection.
    z = label.shape[0] // 2
    base = _norm_to_uint8(image[0, z])
    gt_overlay = _overlay(base, label[z])
    pr_overlay = _overlay(base, pred[z])

    gt_path = fig_dir / f"{stem}_gt.png"
    pr_path = fig_dir / f"{stem}_pred.png"
    Image.fromarray(gt_overlay).save(gt_path)
    Image.fromarray(pr_overlay).save(pr_path)

    axial_path = view_dir / f"{stem}_axial.png"
    sagittal_path = view_dir / f"{stem}_sagittal.png"
    coronal_path = view_dir / f"{stem}_coronal.png"
    _save_three_view(image=image, label=label, pred=pred, out_axial=axial_path, out_sagittal=sagittal_path, out_coronal=coronal_path)

    entropy_path = entropy_dir / f"{stem}_entropy_axial.png"
    _save_gray_map(entropy[z], entropy_path)

    prob_c1_path = probs_dir / f"{stem}_prob_c1_axial.png"
    prob_c2_path = probs_dir / f"{stem}_prob_c2_axial.png"
    prob_c3_path = probs_dir / f"{stem}_prob_c3_axial.png"
    cls_idxs = [1, 2, 3] if probs.shape[0] >= 4 else list(range(min(3, probs.shape[0])))
    _save_gray_map(probs[cls_idxs[0], z], prob_c1_path)
    _save_gray_map(probs[cls_idxs[1], z], prob_c2_path)
    _save_gray_map(probs[cls_idxs[2], z], prob_c3_path)

    return {
        "case_id": case_id,
        "npz": str(npz_path),
        "gt_png": str(gt_path),
        "pred_png": str(pr_path),
        "axial_png": str(axial_path),
        "sagittal_png": str(sagittal_path),
        "coronal_png": str(coronal_path),
        "entropy_png": str(entropy_path),
        "prob_c1_png": str(prob_c1_path),
        "prob_c2_png": str(prob_c2_path),
        "prob_c3_png": str(prob_c3_path),
    }


def _norm_to_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    y = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    return (y * 255).astype(np.uint8)


def _overlay(gray: np.ndarray, seg: np.ndarray) -> np.ndarray:
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    cmap = {
        1: np.array([255, 80, 80], dtype=np.float32),
        2: np.array([80, 255, 80], dtype=np.float32),
        3: np.array([80, 170, 255], dtype=np.float32),
        4: np.array([80, 170, 255], dtype=np.float32),
    }
    out = rgb.copy()
    alpha = 0.45
    for cls, color in cmap.items():
        m = seg == cls
        out[m] = (1 - alpha) * out[m] + alpha * color
    return out.clip(0, 255).astype(np.uint8)


def _save_gray_map(x: np.ndarray, path: Path) -> None:
    u8 = _norm_to_uint8(x.astype(np.float32))
    Image.fromarray(u8).save(path)


def _softmax_np(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-8)


def _entropy_np(probs: np.ndarray, axis: int = 0) -> np.ndarray:
    p = np.clip(probs.astype(np.float32), 1e-8, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def _save_three_view(
    image: np.ndarray,
    label: np.ndarray,
    pred: np.ndarray,
    out_axial: Path,
    out_sagittal: Path,
    out_coronal: Path,
) -> None:
    z = label.shape[0] // 2
    y = label.shape[1] // 2
    x = label.shape[2] // 2

    axial = _overlay(_norm_to_uint8(image[0, z]), pred[z])
    sagittal = _overlay(_norm_to_uint8(image[0, :, :, x]), pred[:, :, x])
    coronal = _overlay(_norm_to_uint8(image[0, :, y, :]), pred[:, y, :])

    gt_axial = _overlay(_norm_to_uint8(image[0, z]), label[z])
    gt_sagittal = _overlay(_norm_to_uint8(image[0, :, :, x]), label[:, :, x])
    gt_coronal = _overlay(_norm_to_uint8(image[0, :, y, :]), label[:, y, :])

    Image.fromarray(np.concatenate([gt_axial, axial], axis=1)).save(out_axial)
    Image.fromarray(np.concatenate([gt_sagittal, sagittal], axis=1)).save(out_sagittal)
    Image.fromarray(np.concatenate([gt_coronal, coronal], axis=1)).save(out_coronal)


def save_feature_attention_artifacts(
    out_dir: str,
    epoch: int,
    sample_idx: int,
    case_id: str,
    feature_maps: Mapping[str, np.ndarray],
    attention_maps: Mapping[str, np.ndarray],
) -> Dict[str, str]:
    root = Path(out_dir)
    feat_dir = root / "intermediate" / "features"
    attn_dir = root / "intermediate" / "attention"
    feat_dir.mkdir(parents=True, exist_ok=True)
    attn_dir.mkdir(parents=True, exist_ok=True)

    stem = f"epoch_{epoch:03d}_sample_{sample_idx:03d}_{case_id}"
    feat_npz_path = feat_dir / f"{stem}_features.npz"
    attn_npz_path = attn_dir / f"{stem}_attention.npz"
    np.savez_compressed(feat_npz_path, **{k: v.astype(np.float32) for k, v in feature_maps.items()})
    np.savez_compressed(attn_npz_path, **{k: v.astype(np.float32) for k, v in attention_maps.items()})

    feat_png_path = feat_dir / f"{stem}_feature_grid.png"
    attn_png_path = attn_dir / f"{stem}_attention_grid.png"
    _save_named_map_grid(feature_maps, feat_png_path)
    _save_named_map_grid(attention_maps, attn_png_path)
    return {
        "feature_npz": str(feat_npz_path),
        "feature_png": str(feat_png_path),
        "attention_npz": str(attn_npz_path),
        "attention_png": str(attn_png_path),
    }


def _save_named_map_grid(named_maps: Mapping[str, np.ndarray], out_path: Path) -> None:
    if not named_maps:
        return
    entries = []
    max_h = 0
    max_w = 0
    for name, arr in named_maps.items():
        map2d = _to_2d(arr)
        tile = _norm_to_uint8(map2d)
        entries.append((name, tile))
        max_h = max(max_h, tile.shape[0])
        max_w = max(max_w, tile.shape[1])

    tiles = []
    for _, tile in entries:
        tile = _pad_to_shape(tile, target_h=max_h, target_w=max_w)
        title_bar = np.full((16, max_w), 24, dtype=np.uint8)
        text_strip = np.stack([title_bar, title_bar, title_bar], axis=-1)
        tile_rgb = np.stack([tile, tile, tile], axis=-1)
        tiles.append(np.concatenate([text_strip, tile_rgb], axis=0))
    grid = np.concatenate(tiles, axis=1)
    Image.fromarray(grid).save(out_path)


def _to_2d(x: np.ndarray) -> np.ndarray:
    arr = x.astype(np.float32)
    while arr.ndim > 3:
        arr = arr.mean(axis=0)
    if arr.ndim == 3:
        z = arr.shape[0] // 2
        return arr[z]
    if arr.ndim == 2:
        return arr
    return np.expand_dims(arr, axis=0)


def _pad_to_shape(x: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = x.shape
    pad_top = max((target_h - h) // 2, 0)
    pad_bottom = max(target_h - h - pad_top, 0)
    pad_left = max((target_w - w) // 2, 0)
    pad_right = max(target_w - w - pad_left, 0)
    return np.pad(x, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)


def save_prediction_nifti(
    pred: np.ndarray,
    out_dir: str,
    epoch: int,
    case_id: str,
    ref_nifti_path: str,
) -> str:
    root = Path(out_dir) / "intermediate" / "nifti"
    root.mkdir(parents=True, exist_ok=True)
    ref = nib.load(ref_nifti_path)
    pred_out = pred.astype(np.int16).copy()
    pred_out[pred_out == 3] = 4
    out_path = root / f"epoch_{epoch:03d}_{case_id}_pred.nii.gz"
    nib.save(nib.Nifti1Image(pred_out, affine=ref.affine, header=ref.header), str(out_path))
    return str(out_path)


def compute_case_volume_stats(
    pred: np.ndarray,
    label: np.ndarray,
    voxel_volume_mm3: float = 1.0,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for cls in (1, 2, 3):
        gt_vox = int((label == cls).sum())
        pr_vox = int((pred == cls).sum())
        out[f"gt_c{cls}_voxels"] = gt_vox
        out[f"pred_c{cls}_voxels"] = pr_vox
        out[f"diff_c{cls}_voxels"] = pr_vox - gt_vox
        out[f"gt_c{cls}_mm3"] = gt_vox * voxel_volume_mm3
        out[f"pred_c{cls}_mm3"] = pr_vox * voxel_volume_mm3
        out[f"diff_c{cls}_mm3"] = (pr_vox - gt_vox) * voxel_volume_mm3
    out["gt_tumor_voxels"] = int((label > 0).sum())
    out["pred_tumor_voxels"] = int((pred > 0).sum())
    out["diff_tumor_voxels"] = out["pred_tumor_voxels"] - out["gt_tumor_voxels"]
    out["gt_tumor_mm3"] = out["gt_tumor_voxels"] * voxel_volume_mm3
    out["pred_tumor_mm3"] = out["pred_tumor_voxels"] * voxel_volume_mm3
    out["diff_tumor_mm3"] = out["diff_tumor_voxels"] * voxel_volume_mm3
    return out


def write_case_metrics(out_dir: str, epoch: int, rows: Sequence[Mapping[str, object]]) -> Dict[str, str]:
    root = Path(out_dir) / "analysis" / "case_metrics"
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / f"epoch_{epoch:03d}_case_metrics.csv"
    json_path = root / f"epoch_{epoch:03d}_case_metrics.json"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    json_path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"case_metrics_csv": str(csv_path), "case_metrics_json": str(json_path)}


def write_volume_stats(out_dir: str, epoch: int, rows: Sequence[Mapping[str, object]]) -> Dict[str, str]:
    root = Path(out_dir) / "analysis" / "volume_stats"
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / f"epoch_{epoch:03d}_volume_stats.csv"
    json_path = root / f"epoch_{epoch:03d}_volume_stats.json"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    json_path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"volume_stats_csv": str(csv_path), "volume_stats_json": str(json_path)}


def write_worst_cases(
    out_dir: str,
    epoch: int,
    rows: Sequence[Mapping[str, object]],
    top_k: int = 10,
    score_key: str = "dice_mean",
) -> Dict[str, str]:
    root = Path(out_dir) / "analysis" / "worst_cases"
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / f"epoch_{epoch:03d}_worst_cases.csv"
    json_path = root / f"epoch_{epoch:03d}_worst_cases.json"
    ranked = sorted(rows, key=lambda x: float(x.get(score_key, 0.0)))[: max(1, top_k)]
    fieldnames = sorted({k for row in ranked for k in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    json_path.write_text(json.dumps(ranked, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"worst_cases_csv": str(csv_path), "worst_cases_json": str(json_path)}


def write_artifact_manifest(
    out_dir: str,
    epoch: int,
    records: Iterable[Mapping[str, object]],
) -> str:
    root = Path(out_dir) / "analysis" / "manifests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"epoch_{epoch:03d}_artifact_manifest.json"
    path.write_text(json.dumps(list(records), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
