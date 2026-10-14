"""Test inference'ı ve model birleştirme (rank blend).

Eski kernel'in 0.500 almasının sebebi sessiz geri dönüşlerdi: DICOM okunamayınca ya da
checkpoint yüklenemeyince sabit önsel değerler yazılıyordu. Burada kural basit:
**okunamayan vaka oranı eşiği aşarsa ya da model yüklenemezse notebook HATA verir.**
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Dataset

from src.cache import CacheConfig, process_study
from src.config import TARGET_COLS
from src.dataset import KneeCacheDataset, collate_studies


class OnTheFlyDataset(KneeCacheDataset):
    """Test vakalarını cache olmadan doğrudan DICOM'dan okur (aynı ön işleme)."""

    def __init__(self, test_df: pd.DataFrame, series_df: pd.DataFrame, data_dir: Path,
                 cache_cfg: CacheConfig, eval_windows_per_slot: int = 14):
        super().__init__(test_df, cache_dir=Path("."), train=False,
                         eval_windows_per_slot=eval_windows_per_slot, with_targets=False)
        self.groups = dict(tuple(series_df.groupby("StudyInstanceUID")))
        self.data_dir, self.cache_cfg = Path(data_dir), cache_cfg

    def __getitem__(self, i):
        study = self.df.iloc[i]["StudyInstanceUID"]
        rows = self.groups.get(study)
        res = (process_study(study, rows, self.data_dir, "test", self.cache_cfg)
               if rows is not None else {"volumes": [], "slots": [], "meta": {"errors": "no_series"}})
        imgs, slots, pos = [], [], []
        for vol, sid in zip(res["volumes"], res["slots"]):
            w, p = self._windows(torch.from_numpy(vol).float() / 255.0)
            imgs.append(w); pos.append(p)
            slots.append(torch.full((len(w),), int(sid), dtype=torch.long))
        return {
            "study_id": study,
            "images": torch.cat(imgs) if imgs else torch.zeros(0, 3, 1, 1),
            "slots": torch.cat(slots) if slots else torch.zeros(0, dtype=torch.long),
            "pos": torch.cat(pos) if pos else torch.zeros(0),
            "meta": res["meta"],
        }


def _collate_with_meta(batch):
    out = collate_studies(batch)
    out["meta"] = [b["meta"] for b in batch]
    return out


@torch.no_grad()
def predict_test(models: Sequence[torch.nn.Module], test_df: pd.DataFrame, series_df: pd.DataFrame,
                 data_dir: Path, cache_cfg: CacheConfig, device, batch_size: int = 2,
                 num_workers: int = 4, max_fail_frac: float = 0.02) -> (pd.DataFrame, pd.DataFrame):
    """Her model için olasılıkları üretir, fold'ları rank-ortalamasıyla birleştirir.

    Dönüş: (tahmin tablosu, vaka meta tablosu). Başarısız oran > max_fail_frac ise hata.
    """
    ds = OnTheFlyDataset(test_df, series_df, data_dir, cache_cfg)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    collate_fn=_collate_with_meta)
    per_model = [[] for _ in models]
    ids, metas = [], []
    for batch in dl:
        images = batch["images"].to(device)
        slots, pos, mask = batch["slots"].to(device), batch["pos"].to(device), batch["mask"].to(device)
        for k, m in enumerate(models):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = m(images, slots, pos, mask)["logits"]
            per_model[k].append(torch.sigmoid(logits.float()).cpu().numpy())
        ids += batch["study_ids"]
        metas += batch["meta"]

    meta = pd.DataFrame(metas)
    meta["StudyInstanceUID"] = ids
    n_fail = int((meta["n_slots"].fillna(0) == 0).sum()) if "n_slots" in meta else len(meta)
    print(f"Okunamayan vaka: {n_fail}/{len(meta)}")
    if n_fail > max_fail_frac * len(meta):
        raise RuntimeError(f"Çok fazla okunamayan vaka ({n_fail}/{len(meta)}). Sabit tahmine "
                           "düşmek yerine duruyorum; meta tablosundaki 'errors' sütununa bakın.")

    failed = (meta["n_slots"].fillna(0) == 0).values if "n_slots" in meta else np.ones(len(meta), bool)
    frames = []
    for preds in per_model:
        df = pd.DataFrame(np.concatenate(preds), columns=TARGET_COLS)
        df.loc[failed, TARGET_COLS] = np.nan  # okunamayan vaka sıralamaya girmez
        df.insert(0, "StudyInstanceUID", ids)
        frames.append(df)
    blended = rank_blend(frames)
    blended[TARGET_COLS] = blended[TARGET_COLS].fillna(0.5)  # nötr orta sıra
    return blended, meta


def rank_blend(frames: List[pd.DataFrame], weights: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """Tahminleri bulgu bazında sıralamaya çevirip ağırlıklı ortalar (0-1 aralığında).

    AUC yalnızca sıralamaya baktığı için olasılık yerine sıra ortalamak, ölçeği farklı
    modelleri adil biçimde birleştirir.
    """
    weights = np.asarray(weights if weights is not None else [1.0] * len(frames), dtype=float)
    weights = weights / weights.sum()
    base = frames[0][["StudyInstanceUID"]].copy()
    for c in TARGET_COLS:
        acc = np.zeros(len(base))
        for w, f in zip(weights, frames):
            v = f.set_index("StudyInstanceUID").loc[base["StudyInstanceUID"], c].values.astype(float)
            r = np.full(len(v), np.nan)
            ok = ~np.isnan(v)
            if ok.any():
                r[ok] = rankdata(v[ok]) / ok.sum()
            acc += w * r  # NaN (okunamayan vaka) NaN olarak kalır
        base[c] = acc
    return base


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Yarışma formatı kontrolleri + sabit tahmin uyarısı."""
    assert list(sub.columns) == ["StudyInstanceUID"] + TARGET_COLS, "Sütun sırası yanlış"
    assert set(sub.StudyInstanceUID) == set(sample.StudyInstanceUID), "Vaka kümesi farklı"
    assert not sub[TARGET_COLS].isna().any().any(), "NaN var"
    assert ((sub[TARGET_COLS] >= 0) & (sub[TARGET_COLS] <= 1)).all().all(), "0-1 dışı değer"
    if len(sub) > 1 and (sub[TARGET_COLS].nunique() == 1).all():
        raise RuntimeError("Tüm vakalara aynı tahmin yazılmış: model çalışmamış olabilir (AUC=0.5).")
