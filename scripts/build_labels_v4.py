"""Collect the out-of-fold predictions of the five Raptor-80 v3 folds from Kaggle and build labels v4.

Outputs: data/external/oof/raptor80_oof_v3.csv, data/train_labels_v4.csv
Usage:   python scripts/build_labels_v4.py   (all five fold datasets must be uploaded)
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DATA_DIR, KAGGLE_API_TOKEN, TARGET_COLS  # noqa: E402
from src.labels import build_training_labels_v4  # noqa: E402

os.environ.setdefault("KAGGLE_API_TOKEN", KAGGLE_API_TOKEN or "")
tmp = Path(tempfile.mkdtemp())
parts = []
for k in range(5):
    tag = f"kadirov80_lbl3_coatnet_f{k}"
    f = f"raptor_oof_{tag}_fold{k}.npz"
    subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download",
                    f"kadirov0/rsna-knee-srv-{tag.replace('_', '-')}", "-f", f, "-p", str(tmp)], capture_output=True)
    if not (tmp / f).exists():
        sys.exit(f"fold {k} OOF not found ({f}); wait until all five folds are uploaded")
    z = np.load(tmp / f, allow_pickle=True)
    parts.append(pd.DataFrame(z["pred"], columns=TARGET_COLS,
                              index=pd.Index(z["ids"].astype(str), name="StudyInstanceUID")))
oof = pd.concat(parts)
assert not oof.index.duplicated().any(), "folds overlap"
out = DATA_DIR / "external" / "oof" / "raptor80_oof_v3.csv"
oof.reset_index().to_csv(out, index=False)
print("Raptor-80 OOF:", oof.shape, "->", out)
v4 = build_training_labels_v4()
v4.to_csv(DATA_DIR / "train_labels_v4.csv", index=False)
v3 = pd.read_csv(DATA_DIR / "train_labels_v3.csv").set_index("StudyInstanceUID")
d = (v4.set_index("StudyInstanceUID")[TARGET_COLS] - v3[TARGET_COLS]).abs().mean()
print("labels v4:", v4.shape, "| mean absolute change vs v3:", d.round(3).to_dict())
