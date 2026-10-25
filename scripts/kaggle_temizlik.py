"""Yarışma sonrası Kaggle temizliği: yalnız final notebook'ları ve bağımlılıklarını bırakır.

Kalanlar: kadirov0/rsna-knee-10sa-blend, miraysudemutlu/rsna-knee-10t-heliosli-09586,
kadirov0/rsna-knee-src, kadirov0/rsna-knee-r80-lbl3-s101, etiket dataset'leri ve RSNA dışı her şey.

Varsayılan: yalnızca listeyi gösterir (hiçbir şey silinmez).
Silmek için:  python scripts/kaggle_temizlik.py --sil
Silme geri alınamaz.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import KAGGLE_API_TOKEN  # noqa: E402

KADIROV_KERNELS = [
    "rsna-knee-inference-submission", "rsna-knee-public-blend-heliosli",
    "rsna-knee-10sb-blend", "rsna-knee-10sc-blend", "rsna-knee-10sd-blend", "rsna-knee-10se-blend",
    "rsna-knee-10sf-blend", "rsna-knee-10sg-blend", "rsna-knee-10sh-blend", "rsna-knee-17-infer-gold-check",
    "rsna-knee-16-corpus80-tar30-s1", "rsna-knee-16-corpus80-tar30-s2",
    "rsna-knee-15-corpus80-tar-s0", "rsna-knee-15-corpus80-tar-s1", "rsna-knee-15-corpus80-tar-s2",
    "rsna-knee-14-corpus80-npz-s0", "rsna-knee-14-corpus80-npz-s1", "rsna-knee-14-corpus80-npz-s2",
    "rsna-knee-11-corpus80-s0", "rsna-knee-11-corpus80-s1", "rsna-knee-11-corpus80-s2",
    "rsna-knee-04-cache", "rsna-knee-05-train", "rsna-knee-05b-train-folds",
    "rsna-knee-07-raptor-retrain", "rsna-knee-07b-raptor-labels-v2", "rsna-knee-07c-raptor-v2-seed7",
    "rsna-knee-08-llm-labels", "rsna-knee-09-geometry-check", "rsna-knee-10-final-blend",
]
KADIROV_DATASETS = [f"rsna-knee-srv-kadirov80-{t}" for t in (
    "lbl3-coatnet-f0", "lbl3-coatnet-f1", "lbl3-coatnet-f2", "lbl3-coatnet-f3", "lbl3-coatnet-f4",
    "lbl3-coatnet-s404", "lbl3-coatnet-s505", "lbl3-dinov2-s202", "lbl3-dinov2-s303",
    "lbl4-coatnet-s202", "lbl4-coatnet-s303", "lbl4-convnext-s202")]
MIRAY_KERNELS = ["rsna-knee-12-raptor80-v2", "rsna-knee-12b-raptor80-v2-s7", "rsna-knee-12c-raptor80-v2-s13",
                 "rsna-knee-12v3-raptor80-lbl3", "rsna-knee-10r-raptor80-only", "rsna-knee-10-final-blend"]
MIRAY_DATASETS = ["rsna-knee-src"]   # kadirov0/rsna-knee-src'nin kullanılmayan kopyası


def miray_token():
    f = ROOT / ".env.teammate"
    for line in f.read_text().splitlines() if f.exists() else []:
        if line.startswith("KAGGLE_API_TOKEN="):
            return line.split("=", 1)[1].strip()
    return None


def run(token, kind, ref, really):
    if not really:
        print(f"  [liste] {kind:8s} {ref}")
        return
    r = subprocess.run([sys.executable, "-m", "kaggle", kind, "delete", ref, "-y"], capture_output=True, text=True,
                       env=dict(os.environ, KAGGLE_API_TOKEN=token))
    ok = r.returncode == 0
    print(f"  {'silindi' if ok else 'HATA   '} {kind:8s} {ref} {'' if ok else (r.stdout + r.stderr).strip()[-120:]}")


if __name__ == "__main__":
    really = "--sil" in sys.argv
    print("SİLME MODU" if really else "Yalnızca liste (silmek için --sil)")
    print(f"kadirov0: {len(KADIROV_KERNELS)} notebook, {len(KADIROV_DATASETS)} dataset")
    for s in KADIROV_KERNELS:
        run(KAGGLE_API_TOKEN, "kernels", f"kadirov0/{s}", really)
    for s in KADIROV_DATASETS:
        run(KAGGLE_API_TOKEN, "datasets", f"kadirov0/{s}", really)
    mt = miray_token()
    print(f"miraysudemutlu: {len(MIRAY_KERNELS)} notebook, {len(MIRAY_DATASETS)} dataset" + ("" if mt else " (token yok, atlandı)"))
    if mt:
        for s in MIRAY_KERNELS:
            run(mt, "kernels", f"miraysudemutlu/{s}", really)
        for s in MIRAY_DATASETS:
            run(mt, "datasets", f"miraysudemutlu/{s}", really)
