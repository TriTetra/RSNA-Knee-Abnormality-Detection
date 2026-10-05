from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

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


def run_baseline():
    print("Loading datasets...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

    # Baseline 1: Prior probabilities (class frequencies) from train set
    priors = train_df[TARGET_COLS].mean()
    print("Computed baseline priors from train labels:")
    for col, val in priors.items():
        print(f"  {col:20s}: {val:.4f}")

    # Build submission DataFrame
    sub = pd.DataFrame({"StudyInstanceUID": test_df["StudyInstanceUID"]})
    for col in TARGET_COLS:
        sub[col] = priors[col]

    out_path = Path(__file__).resolve().parent / "submission.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSaved baseline submission to {out_path}")
    print("Submission preview:")
    print(sub)


if __name__ == "__main__":
    run_baseline()
