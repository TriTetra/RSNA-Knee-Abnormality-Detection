"""Read a DICOM series and turn it into a canonical volume (uint8 [n_slices, size, size]).

Why this is needed: the data set has four traps, and none of them raises an error; they silently break the model.

1. **Slice order:** the file name is the SOP Instance UID; alphabetical order is unrelated to anatomy.
   The true order is the component of `ImagePositionPatient` along the slice normal.
2. **Image orientation:** scanners may store the same plane flipped or mirrored. The row and column
   direction cosines (`ImageOrientationPatient`) are used to bring every series to one orientation.
3. **Physical scale:** the pixel size (mm) varies between series, so we resample to a fixed field of view
   (FOV, mm) rather than a fixed pixel count.
4. **Right / left knee:** medial and lateral findings (menisci, OA, MCL) swap sides in the image with the knee
   side. Right knees are mirrored to look like left knees: horizontal flip on coronal/axial,
   reversed slice order on sagittal.

Patient coordinate system (LPS): +x towards the patient's left, +y posterior, +z towards the head.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None

AXIS_TO_PLANE = {0: "Sagittal", 1: "Coronal", 2: "Axial"}

DECODER_PACKAGES = ["pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "python-gdcm"]


def ensure_decoders(wheel_dirs: Sequence[Path] = ()) -> Dict[str, bool]:
    """Install and verify decoders for compressed DICOM (JPEG Lossless / JPEG 2000).

    Some series are compressed; pydicom cannot open them without extra libraries and raises an error.
    The installer first tries wheels from the given folders (offline), then the internet. The returned dict
    shows which decoder is available; if both are False, compressed series cannot be read.
    """
    import importlib
    import subprocess
    import sys

    def available():
        out = {}
        for mod in ["libjpeg", "openjpeg", "gdcm"]:
            try:
                importlib.import_module(mod)
                out[mod] = True
            except Exception:
                out[mod] = False
        return out

    status = available()
    if all(status.values()):
        return status
    dirs = [str(d) for d in wheel_dirs if Path(d).exists()]
    for d in dirs:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", d,
                        *DECODER_PACKAGES], check=False)
    status = available()
    if not all(status.values()):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *DECODER_PACKAGES], check=False)
        status = available()
    importlib.invalidate_caches()
    return status


@dataclass
class SliceHeader:
    path: Path
    ipp: Optional[np.ndarray]  # ImagePositionPatient (3,)
    iop: Optional[np.ndarray]  # ImageOrientationPatient (6,)
    pixel_spacing: Optional[np.ndarray]  # (row_mm, col_mm)
    rows: int
    cols: int
    instance: int
    laterality: Optional[str]


@dataclass
class SeriesGeometry:
    headers: List[SliceHeader]  # in anatomical order
    plane: str  # plane derived from the geometry
    row_axis: int  # dominant patient axis of the row direction
    col_axis: int
    normal_axis: int
    flip_lr: bool  # horizontal flip for orientation standardisation
    flip_ud: bool
    laterality_tag: Optional[str]
    center_x_mm: Optional[float]  # patient x coordinate of the image centre
    sorted_by_geometry: bool = True
    notes: List[str] = field(default_factory=list)


def _require_pydicom():
    if pydicom is None:
        raise ImportError("pydicom gerekli: pip install pydicom")


def read_header(path: Path) -> SliceHeader:
    _require_pydicom()
    ds = pydicom.dcmread(str(path), stop_before_pixels=True)

    def vec(name, n):
        v = getattr(ds, name, None)
        if v is None or len(v) < n:
            return None
        return np.array([float(x) for x in v[:n]])

    lat = getattr(ds, "Laterality", None) or getattr(ds, "ImageLaterality", None)
    lat = str(lat).strip().upper()[:1] if lat else None
    return SliceHeader(
        path=Path(path),
        ipp=vec("ImagePositionPatient", 3),
        iop=vec("ImageOrientationPatient", 6),
        pixel_spacing=vec("PixelSpacing", 2),
        rows=int(getattr(ds, "Rows", 0) or 0),
        cols=int(getattr(ds, "Columns", 0) or 0),
        instance=int(getattr(ds, "InstanceNumber", 0) or 0),
        laterality=lat if lat in ("L", "R") else None,
    )


def analyze_series(series_dir: Path) -> SeriesGeometry:
    """Read the headers, put the slices in anatomical order and extract the orientation."""
    files = sorted(Path(series_dir).glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"no DICOM files: {series_dir}")
    headers, notes = [], []
    for f in files:
        try:
            headers.append(read_header(f))
        except Exception:  # one broken file must not drop the whole series
            pass
    if len(headers) < len(files):
        notes.append(f"unreadable_files={len(files) - len(headers)}")
    if len(headers) < 3:
        raise ValueError(f"fewer than 3 readable slices: {series_dir}")

    ref = next((h for h in headers if h.iop is not None), None)
    if ref is None:
        # No geometry: sort by InstanceNumber, no orientation fix.
        headers.sort(key=lambda h: h.instance)
        return SeriesGeometry(headers, "Unknown", 0, 1, 2, False, False,
                              _first_lat(headers), None, False, ["iop_missing"])

    r, c = ref.iop[:3], ref.iop[3:]
    n = np.cross(r, c)
    row_axis, col_axis, normal_axis = (int(np.argmax(np.abs(v))) for v in (r, c, n))

    # 1) Slice order: increasing position along the dominant axis of the slice normal
    #    (x: right->left, y: anterior->posterior, z: inferior->superior), the same for every scanner.
    if all(h.ipp is not None for h in headers):
        headers.sort(key=lambda h: h.ipp[normal_axis])
        sorted_geo = True
    else:
        headers.sort(key=lambda h: h.instance)
        sorted_geo = False
        notes.append("ipp_missing_sorted_by_instance")

    # 2) Orientation: row direction towards +x/+y, column direction towards -z (head up) or +y.
    def needs_flip(vec, axis):
        return bool(vec[axis] > 0) if axis == 2 else bool(vec[axis] < 0)

    flip_lr = needs_flip(r, row_axis)
    flip_ud = needs_flip(c, col_axis)

    # 3) Patient x coordinate of the image centre (fallback for laterality).
    center_x = None
    mid = headers[len(headers) // 2]
    if mid.ipp is not None and mid.pixel_spacing is not None and mid.rows and mid.cols:
        center = (mid.ipp + r * mid.pixel_spacing[1] * mid.cols / 2
                  + c * mid.pixel_spacing[0] * mid.rows / 2)
        center_x = float(center[0])

    return SeriesGeometry(
        headers=headers,
        plane=AXIS_TO_PLANE[normal_axis],
        row_axis=row_axis,
        col_axis=col_axis,
        normal_axis=normal_axis,
        flip_lr=flip_lr,
        flip_ud=flip_ud,
        laterality_tag=_first_lat(headers),
        center_x_mm=center_x,
        sorted_by_geometry=sorted_geo,
        notes=notes,
    )


def _first_lat(headers: Sequence[SliceHeader]) -> Optional[str]:
    return next((h.laterality for h in headers if h.laterality), None)


def resolve_laterality(geoms: Sequence[SeriesGeometry], min_offset_mm: float = 10.0) -> Dict:
    """Decide whether a study shows a right or a left knee.

    The DICOM `Laterality` tag wins. Without it, the median x coordinate of the image centres decides:
    x < 0 -> right knee (R). Too close to the midline (|x| < threshold) gives no decision (None) and no mirroring.
    Agreement with the tag where both exist: 98.5%.
    """
    tags = [g.laterality_tag for g in geoms if g.laterality_tag]
    xs = [g.center_x_mm for g in geoms if g.center_x_mm is not None]
    geo = None
    if xs:
        m = float(np.median(xs))
        geo = None if abs(m) < min_offset_mm else ("R" if m < 0 else "L")
    tag = max(set(tags), key=tags.count) if tags else None
    side = tag or geo
    return {"side": side, "tag": tag, "geometry": geo,
            "source": "tag" if tag else ("geometry" if geo else "none")}


def select_indices(n_total: int, n_out: int, keep_frac: float = 0.8) -> np.ndarray:
    """`n_out` evenly spaced slice indices from the central `keep_frac` of the stack."""
    if n_total <= 0:
        return np.zeros(0, dtype=int)
    margin = (1 - keep_frac) / 2
    lo, hi = margin * (n_total - 1), (1 - margin) * (n_total - 1)
    return np.round(np.linspace(lo, hi, n_out)).astype(int)


def _read_pixels(h: SliceHeader) -> np.ndarray:
    ds = pydicom.dcmread(str(h.path))
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + inter
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    if arr.ndim == 3:  # rare: multi-frame / RGB
        arr = arr.mean(axis=-1) if arr.shape[-1] in (3, 4) else arr[arr.shape[0] // 2]
    return arr


def _resample(arr: np.ndarray, spacing: Optional[np.ndarray], size: int, fov_mm: float) -> np.ndarray:
    """Resample to a fixed FOV, then centre-crop / pad -> (size, size)."""
    h, w = arr.shape
    if spacing is None or np.any(spacing <= 0):
        spacing = np.array([fov_mm / max(h, w)] * 2)  # unknown spacing: the whole image is the FOV
    target = fov_mm / size  # mm per pixel
    nh = max(1, int(round(h * spacing[0] / target)))
    nw = max(1, int(round(w * spacing[1] / target)))
    t = torch.from_numpy(np.ascontiguousarray(arr))[None, None]
    t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False,
                      antialias=(nh < h or nw < w))[0, 0]
    out = torch.zeros((size, size), dtype=t.dtype)
    # centre-crop / pad
    sy, sx = max(0, (nh - size) // 2), max(0, (nw - size) // 2)
    dy, dx = max(0, (size - nh) // 2), max(0, (size - nw) // 2)
    ch, cw = min(size, nh), min(size, nw)
    out[dy:dy + ch, dx:dx + cw] = t[sy:sy + ch, sx:sx + cw]
    return out.numpy()


def load_series_volume(
    geom: SeriesGeometry,
    n_slices: int = 16,
    size: int = 224,
    fov_mm: float = 160.0,
    mirror_to_left: bool = False,
    keep_frac: float = 0.8,
    pct: tuple = (0.5, 99.5),
) -> np.ndarray:
    """Return a canonical uint8 volume: [n_slices, size, size].

    With mirror_to_left=True (a right knee) the image is mirrored to the left-knee view.
    """
    _require_pydicom()
    idx = select_indices(len(geom.headers), n_slices, keep_frac)
    vol = np.stack([
        _resample(_read_pixels(geom.headers[i]), geom.headers[i].pixel_spacing, size, fov_mm)
        for i in idx
    ])

    if geom.flip_lr:
        vol = vol[:, :, ::-1]
    if geom.flip_ud:
        vol = vol[:, ::-1, :]

    if mirror_to_left:
        # Mirror along whichever image axis carries the patient's left-right (x) axis.
        if geom.row_axis == 0:
            vol = vol[:, :, ::-1]
        elif geom.col_axis == 0:
            vol = vol[:, ::-1, :]
        elif geom.normal_axis == 0:
            vol = vol[::-1]

    # Series-level intensity normalisation (one window for all slices).
    fg = vol[vol > 0]
    lo, hi = np.percentile(fg if fg.size > 100 else vol, pct)
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    return (vol * 255).round().astype(np.uint8)
