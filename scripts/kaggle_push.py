"""Kodu ve notebook'ları Kaggle'a gönderir (hepsi PRIVATE).

Kullanım:
    python scripts/kaggle_push.py src                 # src/ + train_labels.csv -> dataset
    python scripts/kaggle_push.py kernel 04           # 04_cache_kaggle.ipynb'i çalıştırmak üzere gönder
    python scripts/kaggle_push.py kernel 05
    python scripts/kaggle_push.py kernel 06
    python scripts/kaggle_push.py kernel 07           # Raptor reçetesi + bizim etiketler
    python scripts/kaggle_push.py kernel 08           # LLM ile rapor etiketleme
    python scripts/kaggle_push.py status 04           # kernel durumunu göster

Kernel'ler birbirinin çıktısını `kernel_sources` ile bağlar:
    04 (cache, CPU) -> 05 (eğitim, GPU) -> 06 (submission, GPU, internet kapalı)
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import KAGGLE_API_TOKEN, validate_config  # noqa: E402

COMPETITION = "rsna-knee-abnormality-detection"
BUILD = ROOT / "build" / "kaggle"


def user() -> str:
    out = subprocess.run([sys.executable, "-m", "kaggle", "config", "view"], capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if "username" in line:
            return line.split(":")[-1].strip()
    raise RuntimeError("Kaggle kullanıcı adı okunamadı")


def kaggle(*args):
    validate_config()
    print(">> kaggle", *args)
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], text=True)
    if r.returncode:
        raise SystemExit(r.returncode)


def push_src():
    u = user()
    d = BUILD / "rsna-knee-src"
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(ROOT / "src", d / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "third_party", d / "third_party", ignore=shutil.ignore_patterns("__pycache__"))
    for f in (ROOT / "data").glob("train_labels*.csv"):  # etiket sürümleri (v1, v2, ...)
        shutil.copy(f, d / f.name)
    for f in (ROOT / "data").glob("llm_codes_partial*.csv"):  # 08'in kaldığı yerden devamı için
        shutil.copy(f, d / f.name)
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": "rsna-knee-src", "id": f"{u}/rsna-knee-src",
        "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    exists = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "status", f"{u}/rsna-knee-src"],
                            capture_output=True, text=True).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(d), "-m", "update src", "--dir-mode", "zip")
    else:
        kaggle("datasets", "create", "-p", str(d), "--dir-mode", "zip")


KERNELS = {
    "04": dict(file="04_cache_kaggle.ipynb", slug="rsna-knee-04-cache", gpu=False, internet=True,
               kernels=[]),
    "05": dict(file="05_egitim_kaggle.ipynb", slug="rsna-knee-05-train", gpu=True, internet=True,
               kernels=["rsna-knee-04-cache"]),
    "06": dict(file="06_submission_kaggle.ipynb", slug="rsna-knee-06-submission", gpu=True, internet=False,
               kernels=["rsna-knee-05-train"]),
    "07": dict(file="07_raptor_yeniden_egitim_kaggle.ipynb", slug="rsna-knee-07-raptor-retrain", gpu=True,
               internet=True, kernels=[],
               datasets=["dreaddevelopment/knee-raptor-corpus", "dreaddevelopment/knee-raptor-corpus-ext"]),
    "07b": dict(file="07b_raptor_etiket_v2_kaggle.ipynb", slug="rsna-knee-07b-raptor-labels-v2", gpu=True,
                internet=True, kernels=[],
                datasets=["dreaddevelopment/knee-raptor-corpus", "dreaddevelopment/knee-raptor-corpus-ext"]),
    "09": dict(file="09_raptor_geometri_kontrol_kaggle.ipynb", slug="rsna-knee-09-geometry-check", gpu=False,
               internet=False, kernels=[], datasets=["dreaddevelopment/knee-raptor-corpus"]),
    "10": dict(file="10_final_blend_kaggle.ipynb", slug="rsna-knee-10-final-blend", gpu=True,
               internet=False, kernels=["rsna-knee-07-raptor-retrain"],
               datasets=['heliosli/rsna-knee-compact-r7cnxfsss', 'nartaa/rsna-knee-publication-swa-weights-20261007', 'heliosli/rsna-orthofoundation-fold0-epoch7-20261004']),
    "08": dict(file="08_llm_etiketleme_kaggle.ipynb", slug="rsna-knee-08-llm-labels", gpu=True,
               internet=True, kernels=[]),
}


def push_kernel(key: str):
    u, k = user(), KERNELS[key]
    d = BUILD / k["slug"]
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    shutil.copy(ROOT / "notebooks" / k["file"], d / k["file"])
    meta = {
        "id": f"{u}/{k['slug']}", "title": k["slug"], "code_file": k["file"],
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": k["gpu"], "enable_internet": k["internet"],
        "dataset_sources": [f"{u}/rsna-knee-src"] + k.get("datasets", []),
        "competition_sources": [COMPETITION],
        "kernel_sources": [f"{u}/{s}" for s in k["kernels"]],
    }
    if k["gpu"]:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    kaggle("kernels", "push", "-p", str(d))


def status(key: str):
    kaggle("kernels", "status", f"{user()}/{KERNELS[key]['slug']}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "src":
        push_src()
    elif cmd == "kernel":
        push_kernel(sys.argv[2])
    elif cmd == "status":
        status(sys.argv[2])
    else:
        print(__doc__)
