"""
Image layout similarity: pixel SSIM, edge-structure SSIM (style-robust), and optional position-based.
Used by tests (including tests/report_similarity.py).
"""
from __future__ import annotations

from pathlib import Path

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage import filters
    import numpy as np
    from PIL import Image
    _SSIM_AVAILABLE = True
except ImportError:
    _SSIM_AVAILABLE = False
    filters = None


def _load_grayscale_uint8(path: Path, size: tuple[int, int] | None = None) -> "np.ndarray":
    """Load image as grayscale uint8, optionally resized."""
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    if size and (arr.shape[0], arr.shape[1]) != size:
        img = img.resize((size[1], size[0]), Image.Resampling.LANCZOS)
        arr = np.array(img, dtype=np.uint8)
    return arr


def layout_similarity(
    path_a: Path,
    path_b: Path,
    common_size: tuple[int, int] = (400, 300),
) -> float:
    """
    Structural similarity (SSIM) between two images after resizing to common_size.
    Returns value in [0, 1] (1 = identical). Sensitive to style (font, stroke).
    """
    if not _SSIM_AVAILABLE:
        raise RuntimeError(
            "scikit-image is required for layout similarity; install with: pip install scikit-image"
        )
    a = _load_grayscale_uint8(Path(path_a), common_size)
    b = _load_grayscale_uint8(Path(path_b), common_size)
    score = ssim(a, b, data_range=255)
    return float(max(0.0, min(1.0, score)))


def layout_similarity_edge(
    path_a: Path,
    path_b: Path,
    common_size: tuple[int, int] = (400, 300),
) -> float:
    """
    SSIM on edge maps: emphasize layout structure (lines, block boundaries), downplay
    handwriting style and font. More robust when comparing hand-drawn vs typeset layout.
    Uses Canny-like strong edges only to reduce noise from texture.
    Returns value in [0, 1] (1 = same structure).
    """
    if not _SSIM_AVAILABLE or filters is None:
        raise RuntimeError("scikit-image is required for edge similarity")
    a = _load_grayscale_uint8(Path(path_a), common_size)
    b = _load_grayscale_uint8(Path(path_b), common_size)
    # Strong edges only (Canny-style): smooth then sobel, normalize to reduce texture
    from skimage import exposure
    ea = filters.sobel(filters.gaussian(a, sigma=1.5))
    eb = filters.sobel(filters.gaussian(b, sigma=1.5))
    ea = exposure.rescale_intensity(ea, out_range=(0, 255))
    eb = exposure.rescale_intensity(eb, out_range=(0, 255))
    ea_u8 = np.clip(ea, 0, 255).astype(np.uint8)
    eb_u8 = np.clip(eb, 0, 255).astype(np.uint8)
    score = ssim(ea_u8, eb_u8, data_range=255)
    return float(max(0.0, min(1.0, score)))


def layout_similarity_position(
    positions_a: list[tuple[float, float]],
    positions_b: list[tuple[float, float]],
    *,
    max_mean_dist: float = 0.2,
) -> float:
    """
    Compare two lists of (x_ratio, y_ratio) (0-1). Style-agnostic: only layout matters.
    Both lists are sorted by (y, x); we match by order and compute mean Euclidean distance.
    Returns similarity in [0, 1]: 1 when mean_dist=0, 0 when mean_dist >= max_mean_dist.
    """
    if not positions_a or not positions_b:
        return 0.0
    sa = sorted(positions_a, key=lambda p: (p[1], p[0]))
    sb = sorted(positions_b, key=lambda p: (p[1], p[0]))
    n = min(len(sa), len(sb))
    if n == 0:
        return 0.0
    total = 0.0
    for i in range(n):
        dx = sa[i][0] - sb[i][0]
        dy = sa[i][1] - sb[i][1]
        total += (dx * dx + dy * dy) ** 0.5
    mean_dist = total / n
    if mean_dist >= max_mean_dist:
        return 0.0
    return float(1.0 - mean_dist / max_mean_dist)


def similarity_available() -> bool:
    """Return True if scikit-image is installed and layout_similarity can be used."""
    return _SSIM_AVAILABLE
