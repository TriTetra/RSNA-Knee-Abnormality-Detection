"""80 kesitlik Raptor corpus'unu Kaggle'dan indirip vaka başına sıkıştırılmış .npz'ye çevirir.

Disk kısıtlı bir makinede (ör. 37 GB boş) eğitim yapabilmek için: corpus ham hâliyle ~40 GB,
sıkıştırılmış hâli ~15-20 GB. Parçalar sırayla indirilir, çevrilir ve ham parça hemen silinir;
diskte aynı anda en fazla bir ham parça (~13 GB) bulunur.

Kullanım (sunucuda, Kaggle token'ı KAGGLE_API_TOKEN ortam değişkeninde):
    python scripts/prepare_corpus_npz.py --out ~/rsna/corpus80_npz --tmp ~/rsna/tmp

Sonra eğitim:
    RAPTOR_NPZ_DIR=~/rsna/corpus80_npz RAPTOR_INPUT_ROOT=~/rsna/inputs python third_party/raptor/train_knee.py ...
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

OWNER = os.environ.get("KAGGLE_SOURCE_OWNER", "kadirov0")


def kaggle(*args):
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError((r.stdout + r.stderr)[-500:])
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="sıkıştırılmış vaka dosyalarının klasörü")
    ap.add_argument("--tmp", required=True, help="ham parçaların geçici indirileceği klasör")
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16, help="paralel sıkıştırma iş parçacığı")
    a = ap.parse_args()
    out, tmp = Path(a.out).expanduser(), Path(a.tmp).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    masks_all, ids_all, row = [], [], 0
    for s in range(a.shards):
        t0 = time.time()
        d = tmp / f"s{s}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        print(f"[parça {s}] indiriliyor…", flush=True)
        kaggle("kernels", "output", f"{OWNER}/rsna-knee-11-corpus80-s{s}", "-p", str(d),
               "--page-size", "200", "--file-pattern", r"corpus_.*\.npy$")
        vols = np.load(next(d.rglob(f"corpus_vols_s{s:02d}.npy")), mmap_mode="r")
        masks = np.load(next(d.rglob(f"corpus_masks_s{s:02d}.npy")))
        ids = np.load(next(d.rglob(f"corpus_ids_s{s:02d}.npy")), allow_pickle=True).astype(str)
        assert len(vols) == len(masks) == len(ids)
        print(f"[parça {s}] {len(vols)} vaka, {vols.shape} | indirme {(time.time() - t0) / 60:.1f} dk", flush=True)

        def save(i, base=row):
            p = out / f"{base + i:05d}.npz"
            if not p.exists():
                np.savez_compressed(p, v=np.asarray(vols[i]))

        with ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(save, range(len(vols))))
        masks_all.append(masks)
        ids_all.append(ids)
        row += len(vols)
        del vols
        shutil.rmtree(d)  # ham parçayı hemen sil
        print(f"[parça {s}] tamam | {(time.time() - t0) / 60:.1f} dk | npz toplam "
              f"{sum(f.stat().st_size for f in out.glob('*.npz')) / 1e9:.1f} GB", flush=True)

    np.save(out / "masks.npy", np.concatenate(masks_all))
    np.save(out / "ids.npy", np.concatenate(ids_all).astype(object), allow_pickle=True)
    print(f"Bitti: {row} vaka -> {out}")


if __name__ == "__main__":
    main()
