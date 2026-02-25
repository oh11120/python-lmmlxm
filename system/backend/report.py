from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np


def _volume_stats(seg_path: str) -> Dict[str, int]:
    seg = np.asarray(nib.load(seg_path).dataobj, dtype=np.uint8)
    return {
        "et_voxels": int((seg == 4).sum()),
        "ed_voxels": int((seg == 2).sum()),
        "tc_voxels": int(((seg == 1) | (seg == 4)).sum()),
        "wt_voxels": int((seg > 0).sum()),
    }


def generate_pdf_report(
    report_path: str,
    case_id: str,
    seg_path: str,
    preview_images: Dict[str, str],
    backend_name: str,
) -> str:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    stats = _volume_stats(seg_path)

    c = canvas.Canvas(report_path, pagesize=A4)
    w, h = A4
    y = h - 40

    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Brain Tumor MRI Segmentation Report")
    y -= 24
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Case ID: {case_id}")
    y -= 16
    c.drawString(40, y, f"Generated: {datetime.utcnow().isoformat()}Z")
    y -= 16
    c.drawString(40, y, f"Inference backend: {backend_name}")
    y -= 24

    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Tumor Volume Summary (voxels)")
    y -= 18
    c.setFont("Helvetica", 10)
    for k, v in stats.items():
        c.drawString(48, y, f"- {k}: {v}")
        y -= 14

    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Multi-view Overlay")
    y -= 14

    img_w, img_h = 160, 160
    x0 = 40
    for i, name in enumerate(["axial", "sagittal", "coronal"]):
        p = preview_images.get(name)
        if not p:
            continue
        c.drawImage(ImageReader(p), x0 + i * (img_w + 20), y - img_h, width=img_w, height=img_h)
        c.drawString(x0 + i * (img_w + 20), y - img_h - 12, name)

    c.showPage()
    c.save()
    return report_path
