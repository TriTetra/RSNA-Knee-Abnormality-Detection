"""Kodu ve notebook'ları Kaggle'a gönderir (hepsi PRIVATE).

Kullanım:
    python scripts/kaggle_push.py src                 # src/ + train_labels.csv -> dataset
    python scripts/kaggle_push.py kernel 04           # 04_cache_kaggle.ipynb'i çalıştırmak üzere gönder
    python scripts/kaggle_push.py kernel 05
    python scripts/kaggle_push.py kernel 06
    python scripts/kaggle_push.py kernel 07           # Raptor reçetesi + bizim etiketler
    python scripts/kaggle_push.py kernel 08           # LLM ile rapor etiketleme
    python scripts/kaggle_push.py status 04           # kernel durumunu göster

Başka takım üyesinin GPU kotasıyla çalıştırmak (kaynaklar kadirov0'da kalır, paylaşılmış olmalı):
    python scripts/kaggle_push.py kernel 12 --token-file .env.teammate --as TAKIM_ARKADASI_KULLANICI_ADI

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

# Başka bir takım üyesinin hesabıyla çalıştırmak için: --token-file .env.teammate
# (dosyada KAGGLE_API_TOKEN=...). Kaynaklar (dataset, corpus kernel'leri) varsayılan olarak
# SOURCE_OWNER hesabından okunur; kernel ise token sahibinin hesabında çalışır.
import os  # noqa: E402
if "--token-file" in sys.argv:
    _i = sys.argv.index("--token-file")
    for _line in Path(sys.argv[_i + 1]).read_text().splitlines():
        if _line.strip().startswith("KAGGLE_API_TOKEN="):
            os.environ["KAGGLE_API_TOKEN"] = _line.split("=", 1)[1].strip()
    del sys.argv[_i:_i + 2]
PUSH_AS = None
if "--as" in sys.argv:  # kernel'in oluşturulacağı hesap (token sahibi)
    _i = sys.argv.index("--as")
    PUSH_AS = sys.argv[_i + 1]
    del sys.argv[_i:_i + 2]
SOURCE_OWNER = os.environ.get("KAGGLE_SOURCE_OWNER", "kadirov0")      # corpus / model kernel'leri
SRC_OWNER = os.environ.get("KAGGLE_SRC_OWNER", SOURCE_OWNER)           # rsna-knee-src dataset'i

from src.config import KAGGLE_API_TOKEN, validate_config  # noqa: E402

COMPETITION = "rsna-knee-abnormality-detection"
BUILD = ROOT / "build" / "kaggle"


def user() -> str:
    """Token sahibinin kullanıcı adı (kernel bu hesapta çalışır)."""
    if PUSH_AS:
        return PUSH_AS
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        name = api.get_config_value("username") or getattr(api, "config_values", {}).get("username")
        if name:
            return name
    except Exception:
        pass
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
    # yerel GPU sunucusu için eğitim script'i (LF satır sonlarıyla)
    if (ROOT / "data" / "folds_v3_5.json").exists():  # 5 fold bölmesi (Raptor fold eğitimleri)
        shutil.copy(ROOT / "data" / "folds_v3_5.json", d / "folds_v3_5.json")
    for sh in ("server_raptor80.sh", "server_queue.sh"):
        (d / sh).write_bytes((ROOT / "scripts" / sh).read_bytes().replace(bytes([13, 10]), bytes([10])))
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
               internet=False, kernels=["miraysudemutlu/rsna-knee-12-raptor80-v2", "miraysudemutlu/rsna-knee-12b-raptor80-v2-s7"],
               datasets=['heliosli/rsna-knee-compact-r7cnxfsss', 'nartaa/rsna-knee-publication-swa-weights-20261007', 'heliosli/rsna-orthofoundation-fold0-epoch7-20261004']),
    "10v3": dict(file="10v3_blend_raptor80_etiket_v3_kaggle.ipynb", slug="rsna-knee-10v3-blend-r80-lbl3", gpu=True,
                 internet=False, kernels=["miraysudemutlu/rsna-knee-12v3-raptor80-lbl3"],
                 datasets=['heliosli/rsna-knee-compact-r7cnxfsss', 'nartaa/rsna-knee-publication-swa-weights-20261007', 'heliosli/rsna-orthofoundation-fold0-epoch7-20261004']),
    "10r": dict(file="10r_raptor80_tek_kaggle.ipynb", slug="rsna-knee-10r-raptor80-only", gpu=True,
                internet=False, kernels=["miraysudemutlu/rsna-knee-12-raptor80-v2", "miraysudemutlu/rsna-knee-12b-raptor80-v2-s7"],
                datasets=['heliosli/rsna-knee-compact-r7cnxfsss', 'nartaa/rsna-knee-publication-swa-weights-20261007', 'heliosli/rsna-orthofoundation-fold0-epoch7-20261004']),
    "07c": dict(file="07c_raptor_etiket_v2_seed7_kaggle.ipynb", slug="rsna-knee-07c-raptor-v2-seed7", gpu=True,
                internet=True, kernels=[],
                datasets=["dreaddevelopment/knee-raptor-corpus", "dreaddevelopment/knee-raptor-corpus-ext"]),
    "05b": dict(file="05b_egitim_fold1-4_kaggle.ipynb", slug="rsna-knee-05b-train-folds", gpu=True, internet=True,
                kernels=["rsna-knee-04-cache"]),
    "11_0": dict(file="11_corpus80_parca0_kaggle.ipynb", slug="rsna-knee-11-corpus80-s0", gpu=False,
                 internet=False, kernels=[]),
    "11_1": dict(file="11_corpus80_parca1_kaggle.ipynb", slug="rsna-knee-11-corpus80-s1", gpu=False,
                 internet=False, kernels=[]),
    "11_2": dict(file="11_corpus80_parca2_kaggle.ipynb", slug="rsna-knee-11-corpus80-s2", gpu=False,
                 internet=False, kernels=[]),
    "12": dict(file="12_raptor80_etiket_v2_kaggle.ipynb", slug="rsna-knee-12-raptor80-v2", gpu=True,
               internet=True, kernels=["rsna-knee-11-corpus80-s0", "rsna-knee-11-corpus80-s1", "rsna-knee-11-corpus80-s2"]),
    "12b": dict(file="12b_raptor80_etiket_v2_seed7_kaggle.ipynb", slug="rsna-knee-12b-raptor80-v2-s7", gpu=True,
                internet=True, kernels=["rsna-knee-11-corpus80-s0", "rsna-knee-11-corpus80-s1", "rsna-knee-11-corpus80-s2"]),
    "12c": dict(file="12c_raptor80_etiket_v2_seed13_kaggle.ipynb", slug="rsna-knee-12c-raptor80-v2-s13", gpu=True,
                internet=True, kernels=["rsna-knee-11-corpus80-s0", "rsna-knee-11-corpus80-s1", "rsna-knee-11-corpus80-s2"]),
    "14_0": dict(file="14_corpus80_npz_parca0_kaggle.ipynb", slug="rsna-knee-14-corpus80-npz-s0", gpu=False,
                 internet=False, kernels=["rsna-knee-11-corpus80-s0"]),
    "14_1": dict(file="14_corpus80_npz_parca1_kaggle.ipynb", slug="rsna-knee-14-corpus80-npz-s1", gpu=False,
                 internet=False, kernels=["rsna-knee-11-corpus80-s1"]),
    "14_2": dict(file="14_corpus80_npz_parca2_kaggle.ipynb", slug="rsna-knee-14-corpus80-npz-s2", gpu=False,
                 internet=False, kernels=["rsna-knee-11-corpus80-s2"]),
    "12v3": dict(file="12v3_raptor80_etiket_v3_kaggle.ipynb", slug="rsna-knee-12v3-raptor80-lbl3", gpu=True,
                 internet=True, kernels=["rsna-knee-11-corpus80-s0", "rsna-knee-11-corpus80-s1", "rsna-knee-11-corpus80-s2"]),
    "15_0": dict(file="15_corpus80_tar_parca0_kaggle.ipynb", slug="rsna-knee-15-corpus80-tar-s0", gpu=False,
                 internet=False, kernels=["rsna-knee-14-corpus80-npz-s0"]),
    "15_1": dict(file="15_corpus80_tar_parca1_kaggle.ipynb", slug="rsna-knee-15-corpus80-tar-s1", gpu=False,
                 internet=False, kernels=["rsna-knee-14-corpus80-npz-s1"]),
    "15_2": dict(file="15_corpus80_tar_parca2_kaggle.ipynb", slug="rsna-knee-15-corpus80-tar-s2", gpu=False,
                 internet=False, kernels=["rsna-knee-14-corpus80-npz-s2"]),
    "16_0": dict(file="16_corpus80_tar30_parca0_kaggle.ipynb", slug="rsna-knee-16-corpus80-tar30-s0", gpu=False,
                 internet=False, kernels=["rsna-knee-14-corpus80-npz-s0"]),
    "16_1": dict(file="16_corpus80_tar30_parca1_kaggle.ipynb", slug="rsna-knee-16-corpus80-tar30-s1", gpu=False,
                 internet=False, kernels=["rsna-knee-14-corpus80-npz-s1"]),
    "16_2": dict(file="16_corpus80_tar30_parca2_kaggle.ipynb", slug="rsna-knee-16-corpus80-tar30-s2", gpu=False,
                 internet=False, kernels=["rsna-knee-14-corpus80-npz-s2"]),
    "17": dict(file="17_cikarim_gold_kontrol_kaggle.ipynb", slug="rsna-knee-17-infer-gold-check", gpu=True,
               internet=False, kernels=[], datasets=["kadirov0/rsna-knee-r80-lbl3-s101",
                                                     "kadirov0/rsna-knee-srv-kadirov80-lbl3-coatnet-s505"]),
    "10t": dict(file="10t_heliosli_09586_kaggle.ipynb", slug="rsna-knee-10t-heliosli-09586", gpu=True, internet=False,
                kernels=["goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958"],
                datasets=['goodpjw2008/rsna-knee-2-5d-convnext-reader', 'heliosli/rsna-knee-r6-frozen-fe055406-w1',
                          'nartaa/rsna-knee-publication-swa-weights-20261007', 'heliosli/rsna-orthofoundation-fold0-epoch7-20261004']),
    "18": dict(file="18_competition_summary_kaggle.ipynb", slug="rsna-knee-what-moved-the-leaderboard", gpu=False,
               internet=False, kernels=[], datasets=[], no_src=True),
    "08": dict(file="08_llm_etiketleme_kaggle.ipynb", slug="rsna-knee-08-llm-labels", gpu=True,
               internet=True, kernels=[]),
}

# Submission varyantları (scripts/make_blend_variants.py üretir); kadirov0 hesabında çalışır
PUBLIC_DS = ['heliosli/rsna-knee-r6-frozen-fe055406-w1', 'nartaa/rsna-knee-publication-swa-weights-20261007',
             'heliosli/rsna-orthofoundation-fold0-epoch7-20261004']   # heliosli 0.958 tabanı
R80V3_DS = ["kadirov0/rsna-knee-r80-lbl3-s101"] + [f"kadirov0/rsna-knee-srv-kadirov80-lbl3-coatnet-f{k}" for k in range(5)]
VARIANT_KEYS = [f"10s{v}" for v in "abcdefgh"]
EXTRA_DS = {"10se": ["kadirov0/rsna-knee-srv-kadirov80-lbl3-dinov2-s202"],
            "10sg": ["kadirov0/rsna-knee-srv-kadirov80-lbl4-coatnet-s303"],
            "10sh": ["kadirov0/rsna-knee-srv-kadirov80-lbl3-coatnet-s404", "kadirov0/rsna-knee-srv-kadirov80-lbl3-coatnet-s505"]}
for _k in VARIANT_KEYS:
    KERNELS[_k] = dict(file=f"{_k}_blend_kaggle.ipynb", slug=f"rsna-knee-{_k}-blend", gpu=True, internet=False,
                       kernels=[], datasets=PUBLIC_DS + R80V3_DS[:1] + EXTRA_DS.get(_k, []))


def push_kernel(key: str):
    u, k = user(), KERNELS[key]
    d = BUILD / k["slug"]
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    src_nb = next((ROOT / "notebooks").rglob(k["file"]))   # notebook'lar alt klasörlerde (raptor/, submission/ …)
    shutil.copy(src_nb, d / k["file"])
    meta = {
        "id": f"{u}/{k['slug']}", "title": k["slug"], "code_file": k["file"],
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": k["gpu"], "enable_internet": k["internet"],
        "dataset_sources": ([] if k.get("no_src") else [f"{SRC_OWNER}/rsna-knee-src"]) + k.get("datasets", []),
        "competition_sources": [COMPETITION],
        # "sahip/slug" biçimindeki kaynaklar olduğu gibi, diğerleri SOURCE_OWNER'dan
        "kernel_sources": [s if "/" in s else f"{SOURCE_OWNER}/{s}" for s in k["kernels"]],
    }
    if k["gpu"]:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"kernel: {u}/{k['slug']} | src: {SRC_OWNER} | kernel kaynakları: {SOURCE_OWNER}")
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
