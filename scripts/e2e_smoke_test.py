from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import requests


def find_case(data_root: Path, case_id: Optional[str]) -> tuple[str, Path]:
    if case_id:
        for grade in ["HGG", "LGG"]:
            cdir = data_root / grade / case_id
            if cdir.exists():
                return case_id, cdir
        raise FileNotFoundError(f"Case not found: {case_id}")

    for grade in ["HGG", "LGG"]:
        grade_dir = data_root / grade
        if not grade_dir.exists():
            continue
        for cdir in sorted(grade_dir.iterdir()):
            if not cdir.is_dir():
                continue
            mods = [
                cdir / f"{cdir.name}_flair.nii",
                cdir / f"{cdir.name}_t1.nii",
                cdir / f"{cdir.name}_t1ce.nii",
                cdir / f"{cdir.name}_t2.nii",
            ]
            if all(p.exists() for p in mods):
                return cdir.name, cdir
    raise RuntimeError("No valid case found under data root")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    p.add_argument("--token", type=str, required=True)
    p.add_argument("--backend", type=str, default="pytorch", choices=["pytorch", "onnx", "tensorrt"])
    p.add_argument("--data-root", type=str, default="data/MICCAI_BraTS_2019_Data_Training")
    p.add_argument("--case-id", type=str, default=None)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--poll-interval", type=float, default=2.5)
    p.add_argument("--output-dir", type=str, default="artifacts/smoke")
    args = p.parse_args()

    base = args.base_url.rstrip("/")
    headers = {"x-api-token": args.token}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_id, case_dir = find_case(Path(args.data_root), args.case_id)
    print(f"Using case: {case_id}")

    files = []
    opened = []
    try:
        for mod in ["flair", "t1", "t1ce", "t2"]:
            fp = case_dir / f"{case_id}_{mod}.nii"
            fobj = fp.open("rb")
            opened.append(fobj)
            files.append(("files", (fp.name, fobj, "application/octet-stream")))

        data = {"case_id": f"smoke_{case_id}", "backend": args.backend}
        r = requests.post(f"{base}/api/tasks", headers=headers, files=files, data=data, timeout=120)
        r.raise_for_status()
        payload = r.json()
        task_id = payload["task_id"]
        print("Task created:", task_id)

        started = time.time()
        last = None
        while True:
            s = requests.get(f"{base}/api/tasks/{task_id}", headers=headers, timeout=60)
            s.raise_for_status()
            status = s.json()
            if status != last:
                print(json.dumps(status, ensure_ascii=False, indent=2))
                last = status

            if status["status"] in {"done", "failed"}:
                break
            if time.time() - started > args.timeout:
                raise TimeoutError(f"Timed out waiting for task {task_id}")
            time.sleep(args.poll_interval)

        if status["status"] != "done":
            raise RuntimeError(f"Task failed: {status.get('message')}" )

        seg_out = out_dir / f"{task_id}_seg.nii.gz"
        pdf_out = out_dir / f"{task_id}_report.pdf"

        seg = requests.get(f"{base}/api/tasks/{task_id}/seg", headers=headers, timeout=120)
        seg.raise_for_status()
        seg_out.write_bytes(seg.content)

        pdf = requests.get(f"{base}/api/tasks/{task_id}/report", headers=headers, timeout=120)
        pdf.raise_for_status()
        pdf_out.write_bytes(pdf.content)

        if seg_out.stat().st_size <= 0:
            raise RuntimeError("Downloaded seg file is empty")
        if pdf_out.stat().st_size <= 0:
            raise RuntimeError("Downloaded report file is empty")

        print("Smoke test passed")
        print("Seg:", seg_out)
        print("PDF:", pdf_out)
    finally:
        for f in opened:
            f.close()


if __name__ == "__main__":
    main()
