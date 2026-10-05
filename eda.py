from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

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

print("=== DATASET SHAPES ===")
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("Sample sub :", sample_sub.shape)

print("\n=== TARGET CLASS DISTRIBUTION (POSITIVE RATIOS) ===")
stats = train_df[TARGET_COLS].mean().sort_values(ascending=False)
for col, val in stats.items():
    print(f"{col:20s}: {val:.4f} ({val*100:.1f}%)")

print("\n=== MISSING REPORT VALUES ===")
print("Missing reports in train:", train_df["Report"].isna().sum())

print("\n=== SAMPLE RADIOLOGY REPORT ===")
print(train_df["Report"].iloc[0][:500])
