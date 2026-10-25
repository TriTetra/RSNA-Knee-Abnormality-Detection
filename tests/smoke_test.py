"""Local end-to-end smoke test: cache -> dataset -> model training -> test inference.

Run:   python tests/smoke_test.py
Needs: the DICOMs of at least one test study under data/test_series.
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cache import CacheConfig, build_cache, load_cached
from src.config import DATA_DIR, EXPERIMENTS_DIR, TARGET_COLS
from src.dataset import KneeCacheDataset, collate_studies
from src.engine import cosine_with_warmup, predict, train_one_epoch
from src.inference import predict_test
from src.model import KneeModel


def main():
    ser = pd.read_csv(DATA_DIR / "test_series.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    local = [s for s in test.StudyInstanceUID if (DATA_DIR / "test_series" / s).exists()]
    assert local, "no DICOM files under data/test_series"
    st = local[0]
    cfg = CacheConfig(n_slices=16, size=128)

    out = EXPERIMENTS_DIR / "smoke_cache"
    shutil.rmtree(out, ignore_errors=True)
    t = time.time()
    meta = build_cache(ser[ser.StudyInstanceUID == st], DATA_DIR, "test", out, cfg, n_workers=1)
    print(meta.T.to_string(), f"\ncache: {time.time() - t:.1f}s")
    vol, slots = load_cached(out, st)
    print("cache hacmi:", vol.shape, "slotlar:", slots)

    lab = pd.DataFrame({"StudyInstanceUID": [st, st]})
    for c in TARGET_COLS:
        lab[c] = np.random.rand(2)
    lab["is_gold"] = [True, False]
    dl = DataLoader(KneeCacheDataset(lab, out, train=True, windows_per_slot=4), batch_size=2,
                    collate_fn=collate_studies)
    b = next(iter(dl))
    print({k: tuple(v.shape) for k, v in b.items() if hasattr(v, "shape")})

    model = KneeModel("resnet18", pretrained=False)
    opt = torch.optim.AdamW(model.parameters(), 1e-4)
    sch = cosine_with_warmup(opt, 1, 4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    print("training loss:", train_one_epoch(model, dl, opt, sch, scaler, torch.device("cpu"), log_every=0))
    ev = DataLoader(KneeCacheDataset(lab, out, train=False), batch_size=2, collate_fn=collate_studies)
    print(predict(model, ev, torch.device("cpu")).iloc[:, :4])

    # Guard test: test studies without local DICOMs must raise an error
    try:
        predict_test([model.eval()], test, ser, DATA_DIR, cfg, torch.device("cpu"), num_workers=0)
        print("UYARI: koruma tetiklenmedi" if len(local) < len(test) else "all studies were read")
    except RuntimeError as e:
        print("GUARD WORKS:", str(e)[:90])
    pred, meta = predict_test([model.eval()], test, ser, DATA_DIR, cfg, torch.device("cpu"),
                              num_workers=0, max_fail_frac=1.0)
    print(meta.reindex(columns=["n_slots", "side", "side_source", "errors"]).to_string())
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
