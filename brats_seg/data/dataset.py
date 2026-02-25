from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import random_elastic_deform, random_flip, random_rotate_scale
from .index import CaseRecord, MODALITIES
from .preprocess import (
    denoise_volume,
    load_nifti,
    n4_bias_field_correction,
    random_crop_3d,
    remap_brats_labels,
    zscore_nonzero,
)


@dataclass
class PreprocessConfig:
    use_denoise: bool = True
    use_n4: bool = True
    median_size: int = 3
    gaussian_sigma: float = 1.0


class BraTSDataset(Dataset):
    def __init__(
        self,
        cases: List[CaseRecord],
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        train: bool = True,
        preprocess_cfg: Optional[PreprocessConfig] = None,
        tumor_bias: float = 0.7,
        base_seed: int = 42,
    ) -> None:
        self.cases = cases
        self.patch_size = patch_size
        self.train = train
        self.preprocess_cfg = preprocess_cfg or PreprocessConfig()
        self.tumor_bias = tumor_bias
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.cases)

    def _load_case(self, case: CaseRecord):
        raw_vols = {m: load_nifti(case.image_paths[m]) for m in MODALITIES}
        n4_mask = (raw_vols["t1ce"] != 0).astype(np.uint8)

        channels = []
        for m in MODALITIES:
            vol = raw_vols[m]
            if self.preprocess_cfg.use_denoise:
                vol = denoise_volume(
                    vol,
                    median_size=self.preprocess_cfg.median_size,
                    gaussian_sigma=self.preprocess_cfg.gaussian_sigma,
                )
            if self.preprocess_cfg.use_n4:
                vol = n4_bias_field_correction(
                    vol,
                    mask=n4_mask,
                    convergence_threshold=1e-6,
                    fail_if_unavailable=True,
                )
            vol = zscore_nonzero(vol)
            channels.append(vol)

        image = np.stack(channels, axis=0).astype(np.float32)
        label = remap_brats_labels(load_nifti(case.label_path)).astype(np.int64)
        return image, label

    def __getitem__(self, idx: int):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        seed = (torch.initial_seed() + idx + worker_id * 100003 + self.base_seed) % (2**32)
        rng = np.random.default_rng(seed)
        image, label = self._load_case(self.cases[idx])

        if self.train:
            image, label = random_crop_3d(image, label, self.patch_size, rng=rng, tumor_bias=self.tumor_bias)
            image, label = random_flip(image, label, rng)
            image, label = random_rotate_scale(image, label, rng)
            image, label = random_elastic_deform(image, label, rng, deformation_coeff=0.05, prob=0.3)

        image_t = torch.from_numpy(image).float()
        label_t = torch.from_numpy(label).long()
        return image_t, label_t
