"""Run the Kaggle CLI with the token from .env (KAGGLE_API_TOKEN).

Usage: python scripts/kaggle_cli.py <any kaggle CLI arguments>
e.g.   python scripts/kaggle_cli.py competitions submissions rsna-knee-abnormality-detection
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import KAGGLE_API_TOKEN, validate_config  # noqa: E402


def main() -> int:
    validate_config()
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN
    return subprocess.run([sys.executable, "-m", "kaggle", *sys.argv[1:]], env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
