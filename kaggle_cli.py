import os
import subprocess
import sys

from config import KAGGLE_API_TOKEN, validate_config


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