import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DATA_DIR, TARGET_COLS

train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

print("=== DATASET SHAPES ===")
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("Sample sub :", sample_sub.shape)

gold_df = train_df[train_df["ACL"].notna()]
print(f"\nGold labeled studies : {len(gold_df)}")
print(f"Unlabeled studies    : {len(train_df) - len(gold_df)}")

print("\n=== TARGET CLASS DISTRIBUTION (58 GOLD STUDIES) ===")
stats = gold_df[TARGET_COLS].mean().sort_values(ascending=False)
for col, val in stats.items():
    print(f"{col:20s}: {val:.4f} ({val*100:.1f}%)")

print("\n=== SAMPLE RADIOLOGY REPORT ===")
print(train_df["Report"].iloc[0][:500])
