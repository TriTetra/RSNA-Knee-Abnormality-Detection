"""Read every study once and write it to disk as a compressed `.npz` (the cache).

The training bottleneck is DICOM reading, not the GPU: one study is ~150 files. Instead of repeating that
every epoch, studies are converted once to canonical volumes and stored.

Slot layout: every study is reduced to at most 6 series.
    0 Sagittal  fluid-sensitive (PD/T2 FS)   1 Sagittal  T1 / non fat-suppressed
    2 Coronal   fluid-sensitive              3 Coronal   T1
    4 Axial     fluid-sensitive              5 Axial     T1
If several series fall into one slot, the one with the most slices is used.
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
    """Pick one series folder per slot from the plane and Fluid_Sensitive columns of the series CSV."""
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
    """Process one study: {'volumes': uint8[k,S,H,W], 'slots': int[k], 'meta': {...}}."""
    slot_dirs = assign_slots(study_series, data_dir, split)
    geoms, errors = {}, []
    for slot, d in slot_dirs.items():
        try:
            geoms[slot] = analyze_series(d)
        except Exception as e:  # broken series: log and skip
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
    """Process all studies in parallel and write `out_dir/<study>.npz` + `out_dir/meta.csv`.

    Existing npz files are skipped, so an interrupted run can be resumed.
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
            print(f"  {i + 1}/{len(jobs)} studies processed", flush=True)
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
    """Return (volumes uint8[k,S,H,W], slots int[k]), or None if the file does not exist."""
    p = Path(cache_dir) / f"{study}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return z["volumes"], z["slots"]
