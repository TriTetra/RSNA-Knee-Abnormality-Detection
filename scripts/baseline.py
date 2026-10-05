import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DATA_DIR, EXPERIMENTS_DIR, TARGET_COLS


def run_baseline():
    print("Loading datasets...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

    # Baseline: Prior probabilities (class frequencies) from train set
    priors = train_df[TARGET_COLS].mean()
    print("Computed baseline priors from train labels:")
    for col, val in priors.items():
        print(f"  {col:20s}: {val:.4f}")

    # Build submission DataFrame
    sub = pd.DataFrame({"StudyInstanceUID": test_df["StudyInstanceUID"]})
    for col in TARGET_COLS:
        sub[col] = priors[col]

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EXPERIMENTS_DIR / "baseline_submission.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSaved baseline submission to {out_path}")
    print("Submission preview:")
    print(sub)


if __name__ == "__main__":
    run_baseline()
