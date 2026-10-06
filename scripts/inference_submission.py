import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DATA_DIR, EXPERIMENTS_DIR, TARGET_COLS
from src.data.dataset import get_dataloader
from src.models.coatnet_attention import create_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and generate Kaggle submission.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained PyTorch model checkpoint (.pt).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet34",
        help="Backbone architecture name (e.g. resnet34, coatnet_rmlp_2_rw_384).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device (cuda or cpu).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(EXPERIMENTS_DIR / "submission.csv"),
        help="Path to output submission CSV.",
    )
    return parser.parse_args()


def run_inference(args):
    print("=======================================================")
    print("  RSNA KNEE: INFERENCE & SUBMISSION PIPELINE")
    print("=======================================================")
    print(f"Device    : {args.device}")
    print(f"Backbone  : {args.backbone}")
    print(f"Checkpoint: {args.checkpoint}")

    # Step 1: Load test metadata
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_series = pd.read_csv(DATA_DIR / "test_series.csv")
    sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

    print(f"\nTest Studies count: {len(test_df)}")
    print(f"Test Series count : {len(test_series)}")

    # Step 2: Initialize test DataLoader
    test_loader = get_dataloader(
        df=test_df,
        series_df=test_series,
        batch_size=1,
        shuffle=False,
        is_test=True,
    )

    # Step 3: Instantiate model
    model = create_model(
        backbone_name=args.backbone,
        checkpoint_path=args.checkpoint,
        pretrained=False,
        device=args.device,
    )
    model.eval()

    # Step 4: Run inference across test studies
    print("\nRunning inference...")
    study_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            study_id = batch["study_ids"][0]
            images = [img.to(args.device) for img in batch["images"]]
            planes = [p.to(args.device) for p in batch["planes"]]

            output = model(images, planes)
            probs = output["probs"][0].cpu().numpy()  # (12,)

            record = {"StudyInstanceUID": study_id}
            for col_idx, col_name in enumerate(TARGET_COLS):
                record[col_name] = float(probs[col_idx])
            study_predictions.append(record)

    # Step 5: Format and validate submission DataFrame
    sub_df = pd.DataFrame(study_predictions)

    # Reorder columns to strictly match sample_submission.csv
    cols_order = ["StudyInstanceUID"] + TARGET_COLS
    sub_df = sub_df[cols_order]

    # Step 6: Sanity Checks (Kaggle Rules)
    print("\n--- SUBMISSION VALIDATION CHECKS ---")
    assert sub_df.shape == sample_sub.shape, (
        f"Shape mismatch! Expected {sample_sub.shape}, got {sub_df.shape}"
    )
    assert (sub_df["StudyInstanceUID"].values == sample_sub["StudyInstanceUID"].values).all(), (
        "StudyInstanceUID order mismatch!"
    )
    assert not sub_df.isna().any().any(), "Submission contains NaN values!"
    assert (sub_df[TARGET_COLS] >= 0.0).all().all() and (sub_df[TARGET_COLS] <= 1.0).all().all(), (
        "Predictions outside [0.0, 1.0] range!"
    )
    print("All validation checks passed successfully!")

    # Step 7: Save output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(out_path, index=False)
    print(f"\nSubmission saved to: {out_path.resolve()}")

    print("\n--- SUBMISSION PREVIEW ---")
    print(sub_df.to_string(index=False))

    print("\n=======================================================")
    print("  KAGGLE CLI SUBMISSION KOMUTU:")
    print("=======================================================")
    print(
        f"python scripts/kaggle_cli.py competitions submit "
        f"-c rsna-knee-abnormality-detection "
        f"-f {out_path} "
        f"-m \"CoAtNet 12-Head Attention Inference\""
    )


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
