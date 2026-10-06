import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import pydicom
except ImportError:
    pydicom = None

from src.config import DATA_DIR, TARGET_COLS

# Anatomical plane encoding
PLANE_MAP = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


def load_dicom_array(dcm_path: Union[str, Path], target_size: int = 384) -> np.ndarray:
    """Read a DICOM file and return a normalized 2D numpy array [0, 1] resized to target_size."""
    if pydicom is None:
        raise ImportError("pydicom is required to read DICOM files. Install with 'pip install pydicom'.")

    dcm = pydicom.dcmread(str(dcm_path))
    pixel_array = dcm.pixel_array.astype(np.float32)

    # Apply slope & intercept if available (intensity calibration)
    slope = getattr(dcm, "RescaleSlope", 1.0)
    intercept = getattr(dcm, "RescaleIntercept", 0.0)
    if slope != 1.0 or intercept != 0.0:
        pixel_array = pixel_array * slope + intercept

    # Handle Invert Photometric Interpretation if needed
    if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixel_array = np.amax(pixel_array) - pixel_array

    # Robust min-max intensity normalization (1st to 99th percentile)
    p_low, p_high = np.percentile(pixel_array, (1, 99))
    if p_high > p_low:
        pixel_array = np.clip(pixel_array, p_low, p_high)
        pixel_array = (pixel_array - p_low) / (p_high - p_low)
    else:
        pixel_array = np.zeros_like(pixel_array)

    # Standardize spatial resolution (H, W) -> (target_size, target_size)
    if pixel_array.shape != (target_size, target_size):
        t = torch.from_numpy(pixel_array).unsqueeze(0).unsqueeze(0)
        t = torch.nn.functional.interpolate(
            t, size=(target_size, target_size), mode="bilinear", align_corners=False
        )
        pixel_array = t.squeeze().numpy()

    return pixel_array.astype(np.float32)


def build_25d_windows(slices: List[np.ndarray], target_size: int = 384) -> np.ndarray:
    """
    Take an ordered list of 2D slices and construct 2.5D RGB triplets:
    [Slice_{i-1}, Slice_i, Slice_{i+1}].
    Returns shape: (num_windows, 3, target_size, target_size)
    """
    if len(slices) < 3:
        base = slices[0] if len(slices) > 0 else np.zeros((target_size, target_size), dtype=np.float32)
        window = np.stack([base, base, base], axis=0)  # (3, H, W)
        return window[np.newaxis, ...]  # (1, 3, H, W)

    windows = []
    for i in range(1, len(slices) - 1):
        triplet = np.stack([slices[i - 1], slices[i], slices[i + 1]], axis=0)  # (3, H, W)
        windows.append(triplet)

    windows = np.array(windows, dtype=np.float32)
    return windows


class KneeStudyDataset(Dataset):
    """
    Study-level PyTorch Dataset for RSNA Knee Abnormality Detection.
    Standardizes a patient study into a stack of 2.5D RGB windows across
    Sagittal, Coronal, and Axial planes.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        series_df: pd.DataFrame,
        data_dir: Path = DATA_DIR,
        target_size: int = 384,
        max_windows_per_study: int = 64,
        is_test: bool = False,
        mock_if_missing: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.series_df = series_df
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.max_windows_per_study = max_windows_per_study
        self.is_test = is_test
        self.mock_if_missing = mock_if_missing

        # Pre-group series by StudyInstanceUID for fast lookup
        self.series_by_study = {
            study_id: group for study_id, group in self.series_df.groupby("StudyInstanceUID")
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        row = self.df.iloc[idx]
        study_id = row["StudyInstanceUID"]

        # Retrieve series for this study
        study_series = self.series_by_study.get(study_id, None)

        all_windows = []
        plane_indices = []

        if study_series is not None:
            for _, srow in study_series.iterrows():
                series_id = srow["SeriesInstanceUID"]
                plane = srow.get("Anatomical_Plane", "Sagittal")
                plane_code = PLANE_MAP.get(plane, 0)

                # Locate DICOM series folder (train or test)
                folder_cand1 = self.data_dir / "train_series" / study_id / series_id
                folder_cand2 = self.data_dir / "test_series" / study_id / series_id
                series_folder = folder_cand1 if folder_cand1.exists() else folder_cand2

                slices = []
                if series_folder.exists():
                    dcm_files = sorted(list(series_folder.glob("*.dcm")))
                    # Trimming peripheral air slices (10% to 90%)
                    n_files = len(dcm_files)
                    start_idx = int(n_files * 0.10)
                    end_idx = max(start_idx + 3, int(n_files * 0.90))
                    core_files = dcm_files[start_idx:end_idx]

                    for dpath in core_files:
                        try:
                            s = load_dicom_array(dpath)
                            slices.append(s)
                        except Exception:
                            continue

                if len(slices) >= 3:
                    windows = build_25d_windows(slices, target_size=self.target_size)
                    all_windows.append(windows)
                    plane_indices.extend([plane_code] * len(windows))

        # Fallback / Mock generator for rapid local development or missing scans
        if len(all_windows) == 0:
            if self.mock_if_missing:
                # Generate a standardized synthetic study tensor
                num_mock = 16
                mock_tensor = np.zeros(
                    (num_mock, 3, self.target_size, self.target_size), dtype=np.float32
                )
                final_windows = torch.tensor(mock_tensor, dtype=torch.float32)
                final_planes = torch.tensor([0, 1, 2] * (num_mock // 3 + 1), dtype=torch.long)[:num_mock]
            else:
                raise FileNotFoundError(f"No valid DICOM slices found for StudyInstanceUID: {study_id}")
        else:
            concatenated = np.concatenate(all_windows, axis=0)  # (total_windows, 3, H, W)
            # Uniform subsampling if windows exceed max_windows_per_study
            if len(concatenated) > self.max_windows_per_study:
                indices = np.linspace(0, len(concatenated) - 1, self.max_windows_per_study, dtype=int)
                concatenated = concatenated[indices]
                plane_indices = [plane_indices[i] for i in indices]

            final_windows = torch.tensor(concatenated, dtype=torch.float32)
            final_planes = torch.tensor(plane_indices, dtype=torch.long)

        output: Dict[str, Union[torch.Tensor, str]] = {
            "study_id": study_id,
            "images": final_windows,  # (N_windows, 3, target_size, target_size)
            "planes": final_planes,   # (N_windows,)
        }

        if not self.is_test:
            targets = row[TARGET_COLS].values.astype(np.float32)
            output["targets"] = torch.tensor(targets, dtype=torch.float32)  # (12,)

        return output


def get_dataloader(
    df: pd.DataFrame,
    series_df: pd.DataFrame,
    batch_size: int = 2,
    shuffle: bool = True,
    is_test: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a standardized DataLoader for KneeStudyDataset."""
    dataset = KneeStudyDataset(df=df, series_df=series_df, is_test=is_test)

    # Custom collate because number of windows can vary slightly per patient
    def collate_fn(batch):
        study_ids = [b["study_id"] for b in batch]
        images = [b["images"] for b in batch]
        planes = [b["planes"] for b in batch]
        result = {"study_ids": study_ids, "images": images, "planes": planes}
        if "targets" in batch[0]:
            targets = torch.stack([b["targets"] for b in batch], dim=0)
            result["targets"] = targets
        return result

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


if __name__ == "__main__":
    print("Testing KneeStudyDataset pipeline...")
    train_annotated_path = DATA_DIR / "train_annotated.csv"
    train_series_path = DATA_DIR / "train_series.csv"

    if train_annotated_path.exists() and train_series_path.exists():
        tdf = pd.read_csv(train_annotated_path).head(4)
        tsdf = pd.read_csv(train_series_path)
        loader = get_dataloader(tdf, tsdf, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        print(f"Batch Study IDs : {batch['study_ids']}")
        print(f"Batch Targets   : {batch['targets'].shape}")
        print(f"Patient 1 Image Tensor: {batch['images'][0].shape}")
        print(f"Patient 1 Plane Tensor: {batch['planes'][0].shape}")
        print("Pipeline verification successful!")
