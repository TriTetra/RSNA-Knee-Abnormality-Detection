"""Training labels: the 58 gold studies + soft labels derived from the radiology reports (LLMs).

`train.csv` has radiologist labels for only 58 studies; the labels of the other 4,349 studies are derived
from the reports. This module

* scores label sources against the 58 gold studies (`audit_sources`),
* averages sources into one soft-label table (`build_training_labels`; v3 and v4 add image-model pseudo labels),
* always keeps the gold studies with their original 0/1 labels and flags them with `is_gold`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import DATA_DIR, TARGET_COLS, find_input

# Labels v1 (2026-10-13): mean of two public sources. Gold macro AUC 0.8927.
SOURCES_V1 = {
    # stevenleehans/rsna-knee-llm-report-labels -> llm_labels_v4_blend.csv
    "steven_v4": "llm_labels_v4_blend.csv",
    # pilkwang/rsna-knee-llm-labels -> report_labels_v2.csv
    "pilkwang_v2": "report_labels_v2.csv",
}

# Labels v2 (2026-10-14): three independent readers. Gold macro AUC 0.9014.
# pilkwang is not added separately: it is already inside JEV HYBRID (it would be counted twice).
SOURCES_V2 = {
    "steven_v4": "llm_labels_v4_blend.csv",
    # github.com/NTejas-1/RSNA-Knee-Abnormality-Detection (MIT): Claude Opus 4.5 labels
    "opus45": "labels_v1_opus45.csv",
    # kaggle.com/datasets/riadmohamed42/jev-knee-labels: JEV HYBRID scores
    "jev_hybrid": "HYBRID_labels_scores.csv",
}

# Labels v3 (2026-10-15): teacher-student. v2 (reports) 50% + ConvNeXt 5-fold OOF (images) 50%.
# Gold macro AUC 0.9014 -> 0.9223 (bootstrap 90% CI [+0.007, +0.035]); leaderboard effect only +0.001.
# OOF: every study is scored by the fold model that did not train on it (05 + 05b, trained on labels v2).
V3_OOF_FILE = "convnext_oof_v2.csv"
V3_OOF_WEIGHT = 0.5

DEFAULT_SOURCES = SOURCES_V2


def build_training_labels_v3(train_csv: Path = DATA_DIR / "train.csv") -> pd.DataFrame:
    """Labels v3: weighted mean of the v2 sources and the ConvNeXt OOF predictions.

    Gold studies always keep the radiologist labels (as in build_training_labels).
    """
    v2 = build_training_labels(SOURCES_V2, train_csv=train_csv).set_index("StudyInstanceUID")
    oof = load_source(V3_OOF_FILE).reindex(v2.index)
    gold = v2["is_gold"]
    v3 = v2.copy()
    mix = (1 - V3_OOF_WEIGHT) * v2[TARGET_COLS] + V3_OOF_WEIGHT * oof[TARGET_COLS].fillna(v2[TARGET_COLS])
    v3.loc[~gold, TARGET_COLS] = mix.loc[~gold].values
    return v3.reset_index()


V4_OOF_FILES = {"convnext": "convnext_oof_v2.csv", "raptor80": "raptor80_oof_v3.csv"}


def build_training_labels_v4(train_csv: Path = DATA_DIR / "train.csv") -> pd.DataFrame:
    """Labels v4 (second teacher-student round): 0.5 · v2 reports + 0.25 · ConvNeXt OOF + 0.25 · Raptor-80 v3 OOF.

    The Raptor-80 OOF comes from the five folds trained on labels v3 (written by scripts/build_labels_v4.py).
    The two image models are different families (r = 0.84), so their mean is less noisy than either one.
    Gold studies always keep the radiologist labels. Leaderboard effect: -0.003 versus v3.
    """
    v2 = build_training_labels(SOURCES_V2, train_csv=train_csv).set_index("StudyInstanceUID")
    cnx = load_source(V4_OOF_FILES["convnext"]).reindex(v2.index)[TARGET_COLS].fillna(v2[TARGET_COLS])
    rap = load_source(V4_OOF_FILES["raptor80"]).reindex(v2.index)[TARGET_COLS].fillna(v2[TARGET_COLS])
    gold = v2["is_gold"]
    v4 = v2.copy()
    mix = 0.5 * v2[TARGET_COLS] + 0.25 * cnx + 0.25 * rap
    v4.loc[~gold, TARGET_COLS] = mix.loc[~gold].values
    return v4.reset_index()


def _locate(file_name: str, search_dirs: Optional[List[Path]] = None) -> Optional[Path]:
    dirs = search_dirs or [DATA_DIR, DATA_DIR / "external", Path("/kaggle/input")]
    for d in dirs:
        if not d.exists():
            continue
        direct = d / file_name
        if direct.exists():
            return direct
        hit = find_input(file_name, str(d))
        if hit is not None:
            return hit
    return None


def load_gold(train_csv: Path = DATA_DIR / "train.csv") -> pd.DataFrame:
    """The 58 gold studies (indexed by StudyInstanceUID, 12 columns)."""
    df = pd.read_csv(train_csv)
    gold = df[df[TARGET_COLS[0]].notna()]
    return gold.set_index("StudyInstanceUID")[TARGET_COLS].astype(float)


def load_source(file_name: str) -> pd.DataFrame:
    path = _locate(file_name)
    if path is None:
        raise FileNotFoundError(f"label source not found: {file_name}")
    df = pd.read_csv(path).rename(columns={"STUDY_UID": "StudyInstanceUID"})  # the opus45 file
    return df.set_index("StudyInstanceUID")[TARGET_COLS].astype(float)


def macro_auc(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, float]:
    """Per-finding AUC and their macro mean (key `macro`) on the shared index."""
    idx = y_true.index.intersection(y_pred.index)
    out = {}
    for c in TARGET_COLS:
        yp_col = y_pred.loc[idx, c]
        yt, yp = y_true.loc[idx, c].values, yp_col.fillna(yp_col.median()).values  # empty cell -> median
        out[c] = float(roc_auc_score(yt > 0.5, yp)) if len(np.unique(yt > 0.5)) > 1 else np.nan
    out["macro"] = float(np.nanmean([out[c] for c in TARGET_COLS]))
    return out


def audit_sources(sources: Dict[str, str] = DEFAULT_SOURCES) -> pd.DataFrame:
    """AUC table of each source on the 58 gold studies (one row per source)."""
    gold = load_gold()
    rows = {name: macro_auc(gold, load_source(f)) for name, f in sources.items()}
    return pd.DataFrame(rows).T[["macro"] + TARGET_COLS]


def build_training_labels(
    sources: Dict[str, str] = DEFAULT_SOURCES,
    weights: Optional[Dict[str, float]] = None,
    train_csv: Path = DATA_DIR / "train.csv",
) -> pd.DataFrame:
    """Weighted mean of the sources + the true labels of the gold studies.

    Returns: StudyInstanceUID, 12 targets (soft, 0-1), is_gold.
    A study missing from one source is filled with the mean of the other sources.
    """
    train = pd.read_csv(train_csv)
    uids = train["StudyInstanceUID"]
    weights = weights or {k: 1.0 for k in sources}

    stack, wsum = 0.0, 0.0
    frames = {k: load_source(f).reindex(uids) for k, f in sources.items()}
    mean_all = pd.concat(frames.values()).groupby(level=0).mean().reindex(uids)
    for k, df in frames.items():
        stack = stack + weights[k] * df.fillna(mean_all)
        wsum += weights[k]
    labels = (stack / wsum).clip(0, 1)

    gold = load_gold(train_csv)
    labels.loc[gold.index, TARGET_COLS] = gold.values
    labels = labels.reset_index().rename(columns={"index": "StudyInstanceUID"})
    labels["is_gold"] = labels["StudyInstanceUID"].isin(gold.index)

    if labels[TARGET_COLS].isna().any().any():
        raise ValueError("empty cells left in the label table; check source coverage")
    return labels
