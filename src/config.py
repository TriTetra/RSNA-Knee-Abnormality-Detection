"""Merkezi yapılandırma: yollar, hedef sütunlar ve ortam (yerel / Kaggle) tespiti.

Aynı kod hem yerelde hem de Kaggle notebook'unda çalışır. Kaggle'da yarışma verisi
`/kaggle/input/...` altına bağlanır; yerelde `data/` klasörü kullanılır.
"""
from pathlib import Path
import os

# --- Yollar -----------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
IS_KAGGLE = Path("/kaggle/input").exists()

# /kaggle/input altında yarışma verisinin ~730 bin DICOM'u var; rglob bunların hepsini dolaşır
# ve dakikalar sürer (05 eğitiminde 15+ dk kayıp). Aşağıdaki yardımcılar bu klasörlere girmez.


def _find_competition_dir() -> Path:
    """`train_series.csv` veya `test_series.csv` içeren yarışma klasörünü bulur."""
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
    """Dosya adı `pattern` (ör. "model_fold*.pt") ile eşleşen tüm dosyalar (DICOM klasörlerine girmez)."""
    import fnmatch
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        out += [Path(d) / f for f in sorted(files) if fnmatch.fnmatch(f, pattern)]
    return out


def find_input(name: str, root: str = "/kaggle/input"):
    """`root` altında adı `name` olan ilk dosyayı/klasörü döndürür (DICOM klasörlerine girmez).

    `name` "third_party/raptor/infer_knee.py" gibi alt yol da olabilir.
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

# --- Kaggle API anahtarı (yalnızca yerelde, .env dosyasından) ----------------
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

# --- 12 hedef bulgu (sample_submission sütun sırası) -------------------------
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
    """Kimliği doğrulanmış bir KaggleApi nesnesi döndürür (yalnızca yerel)."""
    validate_config()
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api
