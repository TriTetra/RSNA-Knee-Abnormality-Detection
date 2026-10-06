import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DATA_DIR, TARGET_COLS


def evaluate_llm_labels(train_df: pd.DataFrame, llm_df: pd.DataFrame, file_name: str) -> dict:
    """Evaluate LLM pseudo-labels against the 58 human gold-standard cases."""
    print(f"\n=======================================================")
    print(f"  DENETIM RAPORU: {file_name}")
    print(f"=======================================================")

    gold_df = train_df[train_df["ACL"].notna()].copy()

    merged = pd.merge(
        gold_df[["StudyInstanceUID"] + TARGET_COLS],
        llm_df[["StudyInstanceUID"] + TARGET_COLS],
        on="StudyInstanceUID",
        suffixes=("_true", "_pred"),
    )

    rows = []
    for col in TARGET_COLS:
        y_true = merged[f"{col}_true"].values
        y_pred = merged[f"{col}_pred"].values

        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_pred)
            f1 = f1_score(y_true, (y_pred >= 0.5).astype(int), zero_division=0)
            rows.append({"Hastalik": col, "ROC_AUC": auc, "F1_Score": f1})
        else:
            rows.append({"Hastalik": col, "ROC_AUC": np.nan, "F1_Score": np.nan})

    metrics_df = pd.DataFrame(rows)
    print(metrics_df.to_string(index=False))

    mean_auc = metrics_df["ROC_AUC"].dropna().mean()
    mean_f1 = metrics_df["F1_Score"].dropna().mean()

    print(f"\n>> Makro Ortalama ROC AUC: {mean_auc:.4f}")
    print(f">> Makro Ortalama F1     : {mean_f1:.4f}")

    return {
        "file_name": file_name,
        "metrics_df": metrics_df,
        "mean_auc": mean_auc,
        "mean_f1": mean_f1,
    }


def create_annotated_training_set(train_df: pd.DataFrame, best_llm_df: pd.DataFrame, out_path: Path):
    """
    Build the final annotated training dataset:
    - 58 gold cases keep their original human expert annotations.
    - 4,349 unlabeled cases are filled with the audited LLM pseudo-labels.
    """
    print(f"\n--- NIHAI EGITIM MATRISI OLUSTURULUYOR ---")

    annotated_df = train_df.copy()
    llm_indexed = best_llm_df.set_index("StudyInstanceUID")

    unlabeled_mask = annotated_df["ACL"].isna()
    unlabeled_count = unlabeled_mask.sum()
    gold_count = (~unlabeled_mask).sum()

    print(f"Korunan Altin Vaka Sayisi     : {gold_count}")
    print(f"Pseudo-Etiketlenen Vaka Sayisi : {unlabeled_count}")

    for idx, row in annotated_df[unlabeled_mask].iterrows():
        study_id = row["StudyInstanceUID"]
        if study_id in llm_indexed.index:
            for col in TARGET_COLS:
                annotated_df.at[idx, col] = llm_indexed.loc[study_id, col]

    annotated_df.to_csv(out_path, index=False)
    print(f"Basariyla kaydedildi: {out_path}")
    print(f"Toplam egitim vakasi: {len(annotated_df)}")
    print(f"Kalan bos etiket sayisi: {annotated_df[TARGET_COLS].isna().sum().sum()}")


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")

    candidates = [
        "llm_labels_full.csv",
        "llm_labels_v2.csv",
        "llm_labels_v4_blend.csv",
    ]

    best_result = None
    best_file = None

    for fname in candidates:
        fpath = DATA_DIR / fname
        if fpath.exists():
            llm_df = pd.read_csv(fpath)
            res = evaluate_llm_labels(train_df, llm_df, fname)
            if best_result is None or res["mean_auc"] > best_result["mean_auc"]:
                best_result = res
                best_file = fname

    print(f"\n>>> EN GUCLU PSEUDO-ETIKET DOSYASI: {best_file} (ROC AUC: {best_result['mean_auc']:.4f})")

    # Winning candidate dataset used for final training matrix
    winning_df = pd.read_csv(DATA_DIR / best_file)
    final_out_path = DATA_DIR / "train_annotated.csv"
    create_annotated_training_set(train_df, winning_df, final_out_path)


if __name__ == "__main__":
    main()
