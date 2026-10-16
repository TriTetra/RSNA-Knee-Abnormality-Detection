"""Eğitim etiketleri: 58 altın vaka + rapordan türetilmiş (LLM) yumuşak etiketler.

`train.csv` içinde yalnızca 58 vakanın radyolog etiketi var; kalan 4.349 vaka için
etiketler raporlardan türetilir. Bu modül:

* farklı etiket kaynaklarını 58 altın vakaya karşı ölçer (`audit_sources`),
* kaynakları ortalayarak tek bir yumuşak etiket tablosu üretir (`build_training_labels`),
* altın vakaları her zaman orijinal (0/1) etiketleriyle tutar ve `is_gold` ile işaretler.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import DATA_DIR, TARGET_COLS, find_input

# Etiket v1 (2026-10-13): iki public kaynağın ortalaması. 58 altın vaka makro AUC 0.8927.
SOURCES_V1 = {
    # stevenleehans/rsna-knee-llm-report-labels -> llm_labels_v4_blend.csv
    "steven_v4": "llm_labels_v4_blend.csv",
    # pilkwang/rsna-knee-llm-labels -> report_labels_v2.csv
    "pilkwang_v2": "report_labels_v2.csv",
}

# Etiket v2 (2026-10-14): üç bağımsız okuyucu. 58 altın vaka makro AUC 0.9014.
# pilkwang ayrıca eklenmez: JEV HYBRID'in içinde zaten var (çift sayım olur).
SOURCES_V2 = {
    "steven_v4": "llm_labels_v4_blend.csv",
    # github.com/NTejas-1/RSNA-Knee-Abnormality-Detection (MIT): Claude Opus 4.5 etiketleri
    "opus45": "labels_v1_opus45.csv",
    # kaggle.com/datasets/riadmohamed42/jev-knee-labels: JEV HYBRID skorları
    "jev_hybrid": "HYBRID_labels_scores.csv",
}

# Etiket v3 (2026-10-15): öğretmen–öğrenci. v2 (rapor) %50 + ConvNeXt 5-fold OOF (görüntü) %50.
# 58 altın vaka makro AUC 0.9014 -> 0.9223 (bootstrap %90 GA [+0.007, +0.035]).
# OOF: her vaka, onu eğitimde görmemiş fold modelinden (05 + 05b, etiket v2 ile eğitildi).
V3_OOF_FILE = "convnext_oof_v2.csv"
V3_OOF_WEIGHT = 0.5

DEFAULT_SOURCES = SOURCES_V2


def build_training_labels_v3(train_csv: Path = DATA_DIR / "train.csv") -> pd.DataFrame:
    """Etiket v3: v2 kaynaklarının ortalaması ile ConvNeXt OOF tahminlerinin ağırlıklı ortalaması.

    Altın vakalar her zaman radyolog etiketiyle kalır (build_training_labels ile aynı).
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
    """Etiket v4 (ikinci tur öğretmen-öğrenci): 0.5 · v2 raporları + 0.25 · ConvNeXt OOF + 0.25 · Raptor-80 v3 OOF.

    Raptor-80 OOF'u etiket v3 ile eğitilen 5 fold'dan gelir (scripts/build_labels_v4.py üretir).
    İki görüntü modeli farklı aileden (r = 0.84) olduğu için ortalamaları tek modelden daha az gürültülüdür.
    Altın vakalar her zaman radyolog etiketiyle kalır.
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
    """58 altın vakayı (StudyInstanceUID indeksli, 12 sütun) döndürür."""
    df = pd.read_csv(train_csv)
    gold = df[df[TARGET_COLS[0]].notna()]
    return gold.set_index("StudyInstanceUID")[TARGET_COLS].astype(float)


def load_source(file_name: str) -> pd.DataFrame:
    path = _locate(file_name)
    if path is None:
        raise FileNotFoundError(f"Etiket kaynağı bulunamadı: {file_name}")
    df = pd.read_csv(path).rename(columns={"STUDY_UID": "StudyInstanceUID"})  # opus45 dosyası
    return df.set_index("StudyInstanceUID")[TARGET_COLS].astype(float)


def macro_auc(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, float]:
    """Ortak indeksler üzerinde bulgu bazlı AUC ve makro ortalama (`macro` anahtarı)."""
    idx = y_true.index.intersection(y_pred.index)
    out = {}
    for c in TARGET_COLS:
        yp_col = y_pred.loc[idx, c]
        yt, yp = y_true.loc[idx, c].values, yp_col.fillna(yp_col.median()).values  # boş hücre -> medyan
        out[c] = float(roc_auc_score(yt > 0.5, yp)) if len(np.unique(yt > 0.5)) > 1 else np.nan
    out["macro"] = float(np.nanmean([out[c] for c in TARGET_COLS]))
    return out


def audit_sources(sources: Dict[str, str] = DEFAULT_SOURCES) -> pd.DataFrame:
    """Her kaynağın 58 altın vakadaki AUC tablosu (satır = kaynak)."""
    gold = load_gold()
    rows = {name: macro_auc(gold, load_source(f)) for name, f in sources.items()}
    return pd.DataFrame(rows).T[["macro"] + TARGET_COLS]


def build_training_labels(
    sources: Dict[str, str] = DEFAULT_SOURCES,
    weights: Optional[Dict[str, float]] = None,
    train_csv: Path = DATA_DIR / "train.csv",
) -> pd.DataFrame:
    """Kaynakların ağırlıklı ortalaması + altın vakaların gerçek etiketleri.

    Dönen tablo: StudyInstanceUID, 12 hedef (0-1 yumuşak), is_gold.
    Bir kaynakta eksik olan vaka, diğer kaynakların ortalamasıyla doldurulur.
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
        raise ValueError("Etiket tablosunda boş değer kaldı; kaynak kapsamını kontrol edin.")
    return labels
