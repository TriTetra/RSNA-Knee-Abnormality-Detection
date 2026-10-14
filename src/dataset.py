"""Cache'teki hacimlerden model girdisi üreten PyTorch Dataset.

Bir vaka -> en fazla 6 slot -> her slottan 2.5D pencereler.
2.5D pencere: ardışık üç kesit [i-1, i, i+1] üç kanal (RGB) gibi verilir; böylece 2D
bir omurga (ImageNet ön eğitimli) komşu kesit bağlamını da görür.

Eğitimde her slottan rastgele `windows_per_slot` pencere seçilir (veri artırma + hız);
değerlendirmede tüm pencereler eşit aralıklı alınır.

UYARI: Yatay çevirme (horizontal flip) KULLANILMAZ. Sağ dizler zaten sola aynalandı;
rastgele çevirme medial ile lateral'i karıştırır ve 5 hedefi bozar.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.cache import NUM_SLOTS, load_cached
from src.config import TARGET_COLS

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _random_affine(vol: torch.Tensor, max_rot: float = 10, scale=(0.9, 1.1), shift: float = 0.06):
    """Bir slotun tüm kesitlerine aynı rastgele dönme/ölçek/kaydırma uygular. vol: [S,H,W]."""
    a = math.radians(np.random.uniform(-max_rot, max_rot))
    s = np.random.uniform(*scale)
    tx, ty = np.random.uniform(-shift, shift, 2)
    theta = torch.tensor([[math.cos(a) / s, -math.sin(a) / s, tx],
                          [math.sin(a) / s, math.cos(a) / s, ty]], dtype=torch.float32)
    grid = F.affine_grid(theta[None].expand(vol.shape[0], -1, -1), (vol.shape[0], 1, *vol.shape[1:]),
                         align_corners=False)
    return F.grid_sample(vol[:, None], grid, align_corners=False, padding_mode="zeros")[:, 0]


class KneeCacheDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cache_dir: Path, train: bool,
                 windows_per_slot: int = 8, eval_windows_per_slot: int = 14,
                 with_targets: bool = True):
        self.df = df.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.train = train
        self.k = windows_per_slot if train else eval_windows_per_slot
        self.with_targets = with_targets

    def __len__(self):
        return len(self.df)

    def _windows(self, vol: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """vol [S,H,W] -> pencereler [k,3,H,W] ve göreli konum [k] (0..1)."""
        S = vol.shape[0]
        centers = np.arange(1, S - 1)
        if self.train and len(centers) > self.k:
            centers = np.sort(np.random.choice(centers, self.k, replace=False))
        elif len(centers) > self.k:
            centers = centers[np.round(np.linspace(0, len(centers) - 1, self.k)).astype(int)]
        win = torch.stack([vol[c - 1:c + 2] for c in centers])
        pos = torch.tensor(centers / max(S - 1, 1), dtype=torch.float32)
        return win, pos

    def __getitem__(self, i: int) -> Dict:
        row = self.df.iloc[i]
        study = row["StudyInstanceUID"]
        item = load_cached(self.cache_dir, study)
        imgs, slots, pos = [], [], []
        if item is not None:
            volumes, slot_ids = item
            for vol, sid in zip(volumes, slot_ids):
                v = torch.from_numpy(vol).float() / 255.0
                if self.train:
                    v = _random_affine(v)
                    v = (v * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.08, 0.08)).clamp(0, 1)
                w, p = self._windows(v)
                imgs.append(w)
                pos.append(p)
                slots.append(torch.full((len(w),), int(sid), dtype=torch.long))
        out = {
            "study_id": study,
            "images": torch.cat(imgs) if imgs else torch.zeros(0, 3, 1, 1),
            "slots": torch.cat(slots) if slots else torch.zeros(0, dtype=torch.long),
            "pos": torch.cat(pos) if pos else torch.zeros(0),
        }
        if self.with_targets:
            out["targets"] = torch.tensor(row[TARGET_COLS].values.astype(np.float32))
            out["is_gold"] = bool(row.get("is_gold", False))
        return out


def collate_studies(batch: List[Dict]) -> Dict:
    """Değişken sayıda pencereyi [B, Nmax, 3, H, W] olarak doldurur + maske."""
    n_max = max(max(len(b["images"]) for b in batch), 1)
    H, W = next((b["images"].shape[-2:] for b in batch if len(b["images"])), (1, 1))
    B = len(batch)
    images = torch.zeros(B, n_max, 3, H, W)
    slots = torch.zeros(B, n_max, dtype=torch.long)
    pos = torch.zeros(B, n_max)
    mask = torch.zeros(B, n_max, dtype=torch.bool)
    for j, b in enumerate(batch):
        n = len(b["images"])
        if n:
            images[j, :n], slots[j, :n], pos[j, :n], mask[j, :n] = b["images"], b["slots"], b["pos"], True
    out = {"study_ids": [b["study_id"] for b in batch], "images": images,
           "slots": slots, "pos": pos, "mask": mask}
    if "targets" in batch[0]:
        out["targets"] = torch.stack([b["targets"] for b in batch])
        out["is_gold"] = torch.tensor([b["is_gold"] for b in batch])
    return out


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """[N,3,H,W] 0-1 aralığı -> ImageNet normalizasyonu."""
    return (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
