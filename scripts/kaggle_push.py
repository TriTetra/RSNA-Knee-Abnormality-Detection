"""Push code and notebooks to Kaggle (everything is created PRIVATE).

Usage:
    python scripts/kaggle_push.py src                 # src/ + third_party/ + label CSVs -> dataset <user>/rsna-knee-src
    python scripts/kaggle_push.py kernel 04           # push notebooks/04_cache_kaggle.ipynb as a kernel and run it
    python scripts/kaggle_push.py status 04           # show the kernel status

Run on a teammate's GPU quota (sources stay on SOURCE_OWNER and must be shared with the teammate):
    python scripts/kaggle_push.py kernel 12 --token-file .env.teammate --as TEAMMATE_USERNAME

Kernels chain their outputs with `kernel_sources`, e.g.
    04 (cache, CPU) -> 05 (training, GPU) -> 06 (submission, GPU, internet off)
    11 (80-slice corpus, CPU) -> 12 / 12v3 (Raptor-80 training, GPU)
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --token-file FILE: read KAGGLE_API_TOKEN=... from FILE (a teammate's token). The kernel then runs on the
# token owner's account, while datasets and source kernels are still read from SOURCE_OWNER.
if "--token-file" in sys.argv:
    _i = sys.argv.index("--token-file")
    for _line in Path(sys.argv[_i + 1]).read_text().splitlines():
        if _line.strip().startswith("KAGGLE_API_TOKEN="):
            os.environ["KAGGLE_API_TOKEN"] = _line.split("=", 1)[1].strip()
    del sys.argv[_i:_i + 2]
PUSH_AS = None
if "--as" in sys.argv:  # account that owns the kernel (the token owner)
    _i = sys.argv.index("--as")
    PUSH_AS = sys.argv[_i + 1]
    del sys.argv[_i:_i + 2]
SOURCE_OWNER = os.environ.get("KAGGLE_SOURCE_OWNER", "kadirov0")      # corpus / model kernels
SRC_OWNER = os.environ.get("KAGGLE_SRC_OWNER", SOURCE_OWNER)           # the rsna-knee-src dataset

from src.config import validate_config  # noqa: E402

COMPETITION = "rsna-knee-abnormality-detection"
BUILD = ROOT / "build" / "kaggle"

# Public model weights used by the final blends
PUBLIC_0958 = ["heliosli/rsna-knee-r6-frozen-fe055406-w1", "nartaa/rsna-knee-publication-swa-weights-20261007",
               "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"]
RAPTOR_CORPUS = ["dreaddevelopment/knee-raptor-corpus", "dreaddevelopment/knee-raptor-corpus-ext"]
CORPUS80 = ["rsna-knee-11-corpus80-s0", "rsna-knee-11-corpus80-s1", "rsna-knee-11-corpus80-s2"]

KERNELS = {
    # main pipeline (own ConvNeXt model)
    "04": dict(file="04_cache_kaggle.ipynb", slug="rsna-knee-04-cache", gpu=False, internet=True, kernels=[]),
    "05": dict(file="05_train_kaggle.ipynb", slug="rsna-knee-05-train", gpu=True, internet=True,
               kernels=["rsna-knee-04-cache"]),
    "05b": dict(file="05b_train_folds1-4_kaggle.ipynb", slug="rsna-knee-05b-train-folds", gpu=True, internet=True,
                kernels=["rsna-knee-04-cache"]),
    "06": dict(file="06_submission_kaggle.ipynb", slug="rsna-knee-06-submission", gpu=True, internet=False,
               kernels=["rsna-knee-05-train"]),
    # Raptor recipe
    "07": dict(file="07_raptor_retrain_labels_v2_kaggle.ipynb", slug="rsna-knee-07b-raptor-labels-v2", gpu=True,
               internet=True, kernels=[], datasets=RAPTOR_CORPUS),
    "09": dict(file="09_raptor_geometry_check_kaggle.ipynb", slug="rsna-knee-09-geometry-check", gpu=False,
               internet=False, kernels=[], datasets=RAPTOR_CORPUS[:1]),
    "12": dict(file="12_raptor80_labels_v2_kaggle.ipynb", slug="rsna-knee-12-raptor80-v2", gpu=True,
               internet=True, kernels=CORPUS80),
    "12v3": dict(file="12v3_raptor80_labels_v3_kaggle.ipynb", slug="rsna-knee-12v3-raptor80-lbl3", gpu=True,
                 internet=True, kernels=CORPUS80),
    "17": dict(file="17_inference_gold_check_kaggle.ipynb", slug="rsna-knee-17-infer-gold-check", gpu=True,
               internet=False, kernels=[], datasets=["kadirov0/rsna-knee-r80-lbl3-s101"]),
    # submissions
    "10": dict(file="10_final_blend_kaggle.ipynb", slug="rsna-knee-10-final-blend", gpu=True, internet=False,
               kernels=["rsna-knee-12-raptor80-v2"],
               datasets=["heliosli/rsna-knee-compact-r7cnxfsss"] + PUBLIC_0958[1:]),
    "10sa": dict(file="10sa_blend_kaggle.ipynb", slug="rsna-knee-10sa-blend", gpu=True, internet=False,
                 kernels=[], datasets=PUBLIC_0958 + ["kadirov0/rsna-knee-r80-lbl3-s101"]),
    "10t": dict(file="10t_heliosli_0958_reader_kaggle.ipynb", slug="rsna-knee-10t-heliosli-09586", gpu=True,
                internet=False, kernels=["goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958"],
                datasets=["goodpjw2008/rsna-knee-2-5d-convnext-reader"] + PUBLIC_0958),
    "18": dict(file="18_competition_summary_kaggle.ipynb", slug="rsna-knee-what-moved-the-leaderboard", gpu=False,
               internet=False, kernels=[], datasets=[], no_src=True),
}
# 80-slice corpus pipeline: 11 (build, 3 shards) -> 14 (npz) -> 16 (30 small tar archives for the GPU server)
for _s in range(3):
    KERNELS[f"11_{_s}"] = dict(file=f"11_corpus80_shard{_s}_kaggle.ipynb", slug=f"rsna-knee-11-corpus80-s{_s}",
                               gpu=False, internet=False, kernels=[])
    KERNELS[f"14_{_s}"] = dict(file=f"14_corpus80_npz_shard{_s}_kaggle.ipynb", slug=f"rsna-knee-14-corpus80-npz-s{_s}",
                               gpu=False, internet=False, kernels=[f"rsna-knee-11-corpus80-s{_s}"])
    KERNELS[f"16_{_s}"] = dict(file=f"16_corpus80_tar30_shard{_s}_kaggle.ipynb",
                               slug=f"rsna-knee-16-corpus80-tar30-s{_s}", gpu=False, internet=False,
                               kernels=[f"rsna-knee-14-corpus80-npz-s{_s}"])


def user() -> str:
    """Username of the token owner (the kernel runs on this account)."""
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
    raise RuntimeError("could not read the Kaggle username")


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
    for f in (ROOT / "data").glob("train_labels*.csv"):  # label versions (v1, v2, v3, v4)
        shutil.copy(f, d / f.name)
    if (ROOT / "data" / "folds_v3_5.json").exists():     # 5-fold split for the Raptor fold runs
        shutil.copy(ROOT / "data" / "folds_v3_5.json", d / "folds_v3_5.json")
    for sh in ("server_raptor80.sh", "server_queue.sh"):  # GPU-server scripts, shipped with LF line endings
        (d / sh).write_bytes((ROOT / "scripts" / "server" / sh).read_bytes().replace(b"\r\n", b"\n"))
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": "rsna-knee-src", "id": f"{u}/rsna-knee-src", "licenses": [{"name": "CC0-1.0"}]}, indent=2))
    exists = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "status", f"{u}/rsna-knee-src"],
                            capture_output=True, text=True).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(d), "-m", "update src", "--dir-mode", "zip")
    else:
        kaggle("datasets", "create", "-p", str(d), "--dir-mode", "zip")


def push_kernel(key: str):
    u, k = user(), KERNELS[key]
    d = BUILD / k["slug"]
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    src_nb = next((ROOT / "notebooks").rglob(k["file"]))   # notebooks live in sub-folders (raptor/, submission/ ...)
    shutil.copy(src_nb, d / k["file"])
    meta = {
        "id": f"{u}/{k['slug']}", "title": k["slug"], "code_file": k["file"],
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": k["gpu"], "enable_internet": k["internet"],
        "dataset_sources": ([] if k.get("no_src") else [f"{SRC_OWNER}/rsna-knee-src"]) + k.get("datasets", []),
        "competition_sources": [COMPETITION],
        # "owner/slug" sources are used as given, bare slugs are read from SOURCE_OWNER
        "kernel_sources": [s if "/" in s else f"{SOURCE_OWNER}/{s}" for s in k["kernels"]],
    }
    if k["gpu"]:
        meta["machine_shape"] = "NvidiaTeslaT4"
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"kernel: {u}/{k['slug']} | src: {SRC_OWNER} | kernel sources: {SOURCE_OWNER}")
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
