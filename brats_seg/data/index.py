from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

MODALITIES = ("flair", "t1", "t1ce", "t2")


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    grade: str
    image_paths: Dict[str, str]
    label_path: str


def discover_cases(data_root: str) -> List[CaseRecord]:
    root = Path(data_root)
    cases: List[CaseRecord] = []
    for grade in ("HGG", "LGG"):
        grade_dir = root / grade
        if not grade_dir.exists():
            continue
        for case_dir in sorted(p for p in grade_dir.iterdir() if p.is_dir()):
            case_id = case_dir.name
            image_paths = {m: str(case_dir / f"{case_id}_{m}.nii") for m in MODALITIES}
            label_path = str(case_dir / f"{case_id}_seg.nii")
            missing = [p for p in list(image_paths.values()) + [label_path] if not Path(p).exists()]
            if missing:
                raise FileNotFoundError(f"Case {case_id} is missing files: {missing}")
            cases.append(
                CaseRecord(
                    case_id=case_id,
                    grade=grade,
                    image_paths=image_paths,
                    label_path=label_path,
                )
            )
    if not cases:
        raise RuntimeError(f"No cases found under {data_root}")
    return cases


def to_serializable(records: List[CaseRecord]) -> List[dict]:
    return [
        {
            "case_id": r.case_id,
            "grade": r.grade,
            "image_paths": r.image_paths,
            "label_path": r.label_path,
        }
        for r in records
    ]
