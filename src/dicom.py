"""DICOM serisini okuyup kanonik bir hacme (uint8 [n_slices, size, size]) dönüştürür.

Neden gerekli? Bu veri setinde dört tuzak var ve hiçbiri hata vermez, sadece modeli bozar:

1. **Kesit sırası:** Dosya adı SOP Instance UID'dir; alfabetik sıra anatomik sırayla
   ilgisizdir. Doğru sıra `ImagePositionPatient`'ın kesit normali yönündeki bileşenidir.
2. **Görüntü yönü:** Farklı cihazlar aynı düzlemi ters/aynalı kaydedebilir. Satır ve sütun
   yön kosinüslerine (`ImageOrientationPatient`) bakarak hepsini tek bir yöne çeviririz.
3. **Fiziksel ölçek:** Piksel boyutu (mm) seriden seriye değişir. Sabit piksel sayısına
   değil sabit bir görüş alanına (FOV, mm) göre yeniden örnekleriz.
4. **Sağ / sol diz:** Medial/lateral bulgular (menisküsler, OA, MCL) görüntüde dizin
   tarafına göre yer değiştirir. Sağ dizleri aynalayarak hepsini "sol diz" görünümüne
   getiririz: koronal/aksiyalde yatay çevirme, sagitalde kesit sırasını ters çevirme.

Hasta koordinat sistemi (LPS): +x hastanın soluna, +y arkaya (posterior), +z başa doğru.
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
    """Sıkıştırılmış DICOM (JPEG Lossless / JPEG 2000) çözücülerini kurar ve doğrular.

    Veri setinde bazı seriler sıkıştırılmış; pydicom bunları ek kütüphane olmadan açamaz ve
    hata verir. Kaggle imajında bu paketler yok. Önce verilen klasörlerdeki wheel'lerden
    (internetsiz) kurmayı, olmazsa internetten kurmayı dener. Dönen sözlük hangi çözücünün
    hazır olduğunu gösterir; ikisi de False ise sıkıştırılmış seriler okunamaz.
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
    headers: List[SliceHeader]  # anatomik sırada
    plane: str  # geometriden çıkarılan düzlem
    row_axis: int  # satır yönünün baskın hasta ekseni
    col_axis: int
    normal_axis: int
    flip_lr: bool  # yön standardizasyonu için yatay çevirme
    flip_ud: bool
    laterality_tag: Optional[str]
    center_x_mm: Optional[float]  # görüntü merkezinin hasta x koordinatı
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
    """Başlıkları okur, kesitleri anatomik sıraya koyar ve yön bilgisini çıkarır."""
    files = sorted(Path(series_dir).glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"DICOM yok: {series_dir}")
    headers, notes = [], []
    for f in files:
        try:
            headers.append(read_header(f))
        except Exception:  # tek bozuk dosya tüm seriyi düşürmesin
            pass
    if len(headers) < len(files):
        notes.append(f"unreadable_files={len(files) - len(headers)}")
    if len(headers) < 3:
        raise ValueError(f"Okunabilir kesit < 3: {series_dir}")

    ref = next((h for h in headers if h.iop is not None), None)
    if ref is None:
        # Geometri yok: InstanceNumber'a göre sırala, yön düzeltmesi yapma.
        headers.sort(key=lambda h: h.instance)
        return SeriesGeometry(headers, "Unknown", 0, 1, 2, False, False,
                              _first_lat(headers), None, False, ["iop_missing"])

    r, c = ref.iop[:3], ref.iop[3:]
    n = np.cross(r, c)
    row_axis, col_axis, normal_axis = (int(np.argmax(np.abs(v))) for v in (r, c, n))

    # 1) Kesit sırası: kesit normalinin baskın ekseni boyunca artan konum.
    #    (x: sağ->sol, y: ön->arka, z: aşağı->yukarı) Cihazdan bağımsız sabit bir yön.
    if all(h.ipp is not None for h in headers):
        headers.sort(key=lambda h: h.ipp[normal_axis])
        sorted_geo = True
    else:
        headers.sort(key=lambda h: h.instance)
        sorted_geo = False
        notes.append("ipp_missing_sorted_by_instance")

    # 2) Yön standardizasyonu: satır yönü +x/+y'ye, sütun yönü -z (baş yukarıda) ya da +y'ye.
    def needs_flip(vec, axis):
        return bool(vec[axis] > 0) if axis == 2 else bool(vec[axis] < 0)

    flip_lr = needs_flip(r, row_axis)
    flip_ud = needs_flip(c, col_axis)

    # 4) Görüntü merkezinin hasta x koordinatı (laterality yedeği için).
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
    """Bir vakanın dizinin sağ mı sol mu olduğunu belirler.

    Öncelik DICOM `Laterality` etiketindedir. Yoksa görüntü merkezinin x koordinatının
    medyanına bakılır: x < 0 -> sağ diz (R). Orta hatta çok yakınsa (|x| < eşik) karar
    verilmez (None) ve aynalama yapılmaz.
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
    """Yığının ortadaki `keep_frac` kısmından eşit aralıklı `n_out` kesit indeksi."""
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
    if arr.ndim == 3:  # nadir: çok kareli / RGB
        arr = arr.mean(axis=-1) if arr.shape[-1] in (3, 4) else arr[arr.shape[0] // 2]
    return arr


def _resample(arr: np.ndarray, spacing: Optional[np.ndarray], size: int, fov_mm: float) -> np.ndarray:
    """Sabit FOV'a göre yeniden örnekle, sonra merkezden kırp / doldur -> (size, size)."""
    h, w = arr.shape
    if spacing is None or np.any(spacing <= 0):
        spacing = np.array([fov_mm / max(h, w)] * 2)  # bilinmiyorsa tüm görüntü = FOV
    target = fov_mm / size  # mm / piksel
    nh = max(1, int(round(h * spacing[0] / target)))
    nw = max(1, int(round(w * spacing[1] / target)))
    t = torch.from_numpy(np.ascontiguousarray(arr))[None, None]
    t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False,
                      antialias=(nh < h or nw < w))[0, 0]
    out = torch.zeros((size, size), dtype=t.dtype)
    # merkezden kırp / doldur
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
    """Kanonik uint8 hacim döndürür: [n_slices, size, size].

    mirror_to_left=True ise (sağ diz) görüntü "sol diz" görünümüne aynalanır.
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
        # Hastanın sol-sağ ekseni (x) hangi eksende ise o eksende aynala.
        if geom.row_axis == 0:
            vol = vol[:, :, ::-1]
        elif geom.col_axis == 0:
            vol = vol[:, ::-1, :]
        elif geom.normal_axis == 0:
            vol = vol[::-1]

    # Seri düzeyinde yoğunluk normalizasyonu (tüm kesitler aynı pencereyle).
    fg = vol[vol > 0]
    lo, hi = np.percentile(fg if fg.size > 100 else vol, pct)
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    return (vol * 255).round().astype(np.uint8)
