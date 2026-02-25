from __future__ import annotations

import random
from typing import Dict, List

from .index import CaseRecord


def _split_one_group(case_ids: List[str], train_ratio: float, val_ratio: float) -> Dict[str, List[str]]:
    n = len(case_ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    train_ids = case_ids[:n_train]
    val_ids = case_ids[n_train : n_train + n_val]
    test_ids = case_ids[n_train + n_val :]
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def stratified_split(
    cases: List[CaseRecord],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[str]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1")

    rng = random.Random(seed)
    grouped: Dict[str, List[str]] = {"HGG": [], "LGG": []}
    for c in cases:
        grouped[c.grade].append(c.case_id)

    parts = {"train": [], "val": [], "test": []}
    for grade, ids in grouped.items():
        ids = ids.copy()
        rng.shuffle(ids)
        local = _split_one_group(ids, train_ratio, val_ratio)
        for k in parts:
            parts[k].extend(local[k])

    for k in parts:
        rng.shuffle(parts[k])
    return parts


def filter_cases(cases: List[CaseRecord], case_ids: List[str]) -> List[CaseRecord]:
    idx = {c.case_id: c for c in cases}
    return [idx[cid] for cid in case_ids]
