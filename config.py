from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR/".env")
KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")


def validate_config() -> None:
    if not KAGGLE_API_TOKEN:
        raise RuntimeError(
            "KAGGLE_API_TOKEN is missing. "
            "Set it in the project. "
        )