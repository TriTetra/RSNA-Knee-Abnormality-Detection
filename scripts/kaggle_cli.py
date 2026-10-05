import os
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import KAGGLE_API_TOKEN, validate_config


def main() -> int:
    validate_config()

    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = KAGGLE_API_TOKEN

    result = subprocess.run(
        [sys.executable, "-m", "kaggle", *sys.argv[1:]],
        env=env,
    )

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())