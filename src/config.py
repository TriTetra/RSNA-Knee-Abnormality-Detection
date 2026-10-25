"""Central configuration: paths, target columns and environment (local / Kaggle) detection.

The same code runs locally and inside Kaggle notebooks. On Kaggle the competition data is mounted under
`/kaggle/input/...`; locally the `data/` folder is used.
"""
from pathlib import Path
import os

# --- Paths -----------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
IS_KAGGLE = Path("/kaggle/input").exists()

# /kaggle/input holds ~730k competition DICOM files; rglob walks all of them and takes minutes
# (it cost 15+ minutes in notebook 05). The helpers below never enter those folders.


def _find_competition_dir() -> Path:
    """Find the competition folder (the one containing `test_series.csv`)."""
    if IS_KAGGLE:
        for cand in [
            Path("/kaggle/input/rsna-knee-abnormality-detection"),
            Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        ]:
            if (cand / "test_series.csv").exists():
                return cand
        hit = find_input("test_series.csv")
        if hit is not None:
            return hit.parent
    return ROOT_DIR / "data"


SKIP_DIRS = {"train_series", "test_series", "train_images", "test_images"}


def find_inputs(pattern: str, root: str = "/kaggle/input"):
    """All files whose name matches `pattern` (e.g. "model_fold*.pt"); never enters the DICOM folders."""
    import fnmatch
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        out += [Path(d) / f for f in sorted(files) if fnmatch.fnmatch(f, pattern)]
    return out


def find_input(name: str, root: str = "/kaggle/input"):
    """First file or folder named `name` under `root`; never enters the DICOM folders.

    `name` may be a relative path such as "third_party/raptor/infer_knee.py".
    """
    parts = Path(name).parts
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        cand = Path(d).joinpath(*parts)
        if cand.exists():
            return cand
    return None


DATA_DIR = _find_competition_dir()

EXPERIMENTS_DIR = Path("/kaggle/working") if IS_KAGGLE else ROOT_DIR / "experiments"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# --- Kaggle API token (local only, from .env) ----------------
KAGGLE_API_TOKEN = None
if not IS_KAGGLE:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT_DIR / ".env")
    except ImportError:
        pass
    KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")
    if KAGGLE_API_TOKEN:
        os.environ["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN

# --- The 12 target findings (sample_submission column order) -------------------------
TARGET_COLS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
NUM_CLASSES = len(TARGET_COLS)


def validate_config() -> None:
    if not KAGGLE_API_TOKEN:
        raise RuntimeError(
            "KAGGLE_API_TOKEN is missing. Please check the .env file in the project root."
        )


def get_kaggle_api():
    """Return an authenticated KaggleApi object (local only)."""
    validate_config()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api
