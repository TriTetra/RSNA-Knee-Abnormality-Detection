"""Her vakayı bir kez okuyup diske sıkıştırılmış `.npz` olarak yazar (önbellek / cache).

Eğitimin darboğazı GPU değil DICOM okumaktır: bir vaka ~150 dosya. Bunu her epoch'ta
tekrarlamak yerine bir kez kanonik hacimlere dönüştürüp saklarız.

Slot (yuva) yapısı: her vaka en fazla 6 seriye indirgenir.
    0 Sagittal  sıvı-duyarlı (PD/T2 FS)   1 Sagittal  T1 / yağ baskısız
    2 Coronal   sıvı-duyarlı              3 Coronal   T1
    4 Axial     sıvı-duyarlı              5 Axial     T1
Aynı slota birden fazla seri düşerse en çok kesiti olan seçilir.
"""
from __future__ import annotations

import json
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.dicom import analyze_series, load_series_volume, resolve_laterality

SLOTS = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Coronal", 0), ("Axial", 1), ("Axial", 0)]
SLOT_NAMES = ["sag_fs", "sag_t1", "cor_fs", "cor_t1", "ax_fs", "ax_t1"]
NUM_SLOTS = len(SLOTS)


@dataclass
class CacheConfig:
    n_slices: int = 16
    size: int = 224
    fov_mm: float = 160.0
    keep_frac: float = 0.8
    mirror_right_knees: bool = True


def find_series_dir(data_dir: Path, split: str, study: str, series: str) -> Optional[Path]:
    for cand in [data_dir / f"{split}_series" / study / series,
                 data_dir / f"{split}_series" / series,
                 data_dir / study / series]:
        if cand.exists():
            return cand
    return None


def assign_slots(study_series: pd.DataFrame, data_dir: Path, split: str) -> Dict[int, Path]:
    """CSV'deki düzlem + Fluid_Sensitive bilgisine göre her slot için bir seri klasörü seçer."""
    chosen: Dict[int, tuple] = {}
    for _, r in study_series.iterrows():
        key = (r["Anatomical_Plane"], int(r["Fluid_Sensitive"]))
        if key not in SLOTS:
            continue
        d = find_series_dir(data_dir, split, r["StudyInstanceUID"], r["SeriesInstanceUID"])
        if d is None:
            continue
        n = len(list(d.glob("*.dcm")))
        slot = SLOTS.index(key)
        if slot not in chosen or n > chosen[slot][1]:
            chosen[slot] = (d, n)
    return {k: v[0] for k, v in chosen.items()}


def process_study(study: str, study_series: pd.DataFrame, data_dir: Path, split: str,
                  cfg: CacheConfig) -> Dict:
    """Bir vakayı işler: {'volumes': uint8[k,S,H,W], 'slots': int[k], 'meta': {...}}."""
    slot_dirs = assign_slots(study_series, data_dir, split)
    geoms, errors = {}, []
    for slot, d in slot_dirs.items():
        try:
            geoms[slot] = analyze_series(d)
        except Exception as e:  # bozuk seri: logla, atla
            errors.append(f"{SLOT_NAMES[slot]}:header:{type(e).__name__}")

    lat = resolve_laterality(list(geoms.values()))
    mirror = cfg.mirror_right_knees and lat["side"] == "R"

    vols, slots = [], []
    for slot, g in sorted(geoms.items()):
        try:
            vols.append(load_series_volume(g, cfg.n_slices, cfg.size, cfg.fov_mm,
                                           mirror_to_left=mirror, keep_frac=cfg.keep_frac))
            slots.append(slot)
        except Exception as e:
            errors.append(f"{SLOT_NAMES[slot]}:pixels:{type(e).__name__}")

    meta = {
        "StudyInstanceUID": study,
        "n_slots": len(slots),
        "slots": ",".join(SLOT_NAMES[s] for s in slots),
        "side": lat["side"], "side_source": lat["source"],
        "side_tag": lat["tag"], "side_geometry": lat["geometry"],
        "mirrored": mirror,
        "plane_mismatch": ",".join(SLOT_NAMES[s] for s, g in geoms.items()
                                   if g.plane != SLOTS[s][0]),
        "errors": ";".join(errors),
    }
    empty = np.zeros((0, cfg.n_slices, cfg.size, cfg.size), np.uint8)
    return {"volumes": np.stack(vols) if vols else empty,
            "slots": np.array(slots, dtype=np.int64), "meta": meta}


def _worker(args):
    study, rows, data_dir, split, cfg, out_dir = args
    out = Path(out_dir) / f"{study}.npz"
    try:
        res = process_study(study, rows, Path(data_dir), split, cfg)
        if len(res["slots"]):
            np.savez_compressed(out, volumes=res["volumes"], slots=res["slots"])
        return res["meta"]
    except Exception:
        return {"StudyInstanceUID": study, "n_slots": 0, "errors": traceback.format_exc()[-300:]}


def build_cache(series_df: pd.DataFrame, data_dir: Path, split: str, out_dir: Path,
                cfg: CacheConfig = CacheConfig(), n_workers: int = 4,
                studies: Optional[List[str]] = None, log_every: int = 200) -> pd.DataFrame:
    """Tüm vakaları paralel işler, `out_dir/<study>.npz` + `out_dir/meta.csv` yazar.

    Zaten var olan npz dosyaları atlanır; böylece yarıda kalan bir çalışma devam ettirilebilir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cache_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    groups = dict(tuple(series_df.groupby("StudyInstanceUID")))
    todo = [s for s in (studies or list(groups)) if not (out_dir / f"{s}.npz").exists()]
    jobs = [(s, groups[s], str(data_dir), split, cfg, str(out_dir)) for s in todo]

    metas = []
    if n_workers <= 1:
        results = map(_worker, jobs)
        ex = None
    else:
        ex = ProcessPoolExecutor(max_workers=n_workers)
        results = ex.map(_worker, jobs, chunksize=4)
    for i, m in enumerate(results):
        metas.append(m)
        if (i + 1) % log_every == 0:
            print(f"  {i + 1}/{len(jobs)} vaka işlendi", flush=True)
    if ex is not None:
        ex.shutdown()

    meta = pd.DataFrame(metas)
    meta_path = out_dir / "meta.csv"
    if meta_path.exists():
        old = pd.read_csv(meta_path)
        meta = pd.concat([old[~old.StudyInstanceUID.isin(meta.StudyInstanceUID)], meta])
    meta.to_csv(meta_path, index=False)
    return meta


def load_cached(cache_dir: Path, study: str):
    """(volumes uint8[k,S,H,W], slots int[k]) döndürür; dosya yoksa None."""
    p = Path(cache_dir) / f"{study}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return z["volumes"], z["slots"]
