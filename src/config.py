from pathlib import Path
import os
from dotenv import load_dotenv

# Path Configuration
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
SCRIPTS_DIR = ROOT_DIR / "scripts"

# Environment & Kaggle Credentials
load_dotenv(ROOT_DIR / ".env")
KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")

if KAGGLE_API_TOKEN:
    os.environ["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN

# 12 Target Abnormality Findings
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


def validate_config() -> None:
    if not KAGGLE_API_TOKEN:
        raise RuntimeError(
            "KAGGLE_API_TOKEN is missing. "
            "Please check the .env file in the project root."
        )


def get_kaggle_api():
    """Return an authenticated KaggleApi instance."""
    validate_config()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api