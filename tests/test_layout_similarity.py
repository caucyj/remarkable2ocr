"""Tests for layout render vs original page image similarity (satisfaction/render_*.png vs pages/page_*.png)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.layout.similarity import layout_similarity, similarity_available

try:
    from PIL import Image
    import numpy as np
    _SSIM_AVAILABLE = True
except ImportError:
    _SSIM_AVAILABLE = False

# Use shared availability when possible
try:
    _SSIM_AVAILABLE = similarity_available()
except Exception:
    pass

# Minimum layout similarity (0~1) required for each page
MIN_LAYOUT_SIMILARITY = 0.95


def _collect_page_pairs(project_dir: Path) -> list[tuple[int, Path, Path]]:
    """Collect (index, page_path, render_path) for each page that has both."""
    pages_dir = project_dir / "pages"
    satisfaction_dir = project_dir / "satisfaction"
    if not pages_dir.is_dir() or not satisfaction_dir.is_dir():
        return []
    pairs: list[tuple[int, Path, Path]] = []
    for p in sorted(pages_dir.glob("page_*.png")):
        try:
            n = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        render_path = satisfaction_dir / f"render_{n}.png"
        if render_path.is_file():
            pairs.append((n, p, render_path))
    return pairs


def test_layout_render_similarity_per_page_identical(tmp_path: Path) -> None:
    """When satisfaction/render_n.png matches pages/page_n.png, similarity >= 95%."""
    if not _SSIM_AVAILABLE:
        pytest.skip("scikit-image not installed")
    # Create identical images so similarity = 1.0
    (tmp_path / "pages").mkdir(parents=True)
    (tmp_path / "satisfaction").mkdir(parents=True)
    from PIL import Image
    import numpy as np
    img = Image.fromarray(np.ones((200, 320), dtype=np.uint8) * 200, mode="L")
    page_path = tmp_path / "pages" / "page_0.png"
    render_path = tmp_path / "satisfaction" / "render_0.png"
    img.save(page_path)
    img.save(render_path)
    sim = layout_similarity(page_path, render_path)
    assert sim >= MIN_LAYOUT_SIMILARITY, f"Similarity {sim:.2%} below {MIN_LAYOUT_SIMILARITY:.0%}"
    assert sim >= 0.99  # identical


def test_layout_render_similarity_fails_with_actual_value(tmp_path: Path) -> None:
    """When images differ, similarity is low; failure path would output actual value (see project_dir test)."""
    if not _SSIM_AVAILABLE:
        pytest.skip("scikit-image not installed")
    (tmp_path / "pages").mkdir(parents=True)
    (tmp_path / "satisfaction").mkdir(parents=True)
    from PIL import Image
    import numpy as np
    orig = Image.fromarray(np.ones((200, 320), dtype=np.uint8) * 200, mode="L")
    orig.save(tmp_path / "pages" / "page_0.png")
    render = Image.fromarray(np.zeros((200, 320), dtype=np.uint8), mode="L")
    render.save(tmp_path / "satisfaction" / "render_0.png")
    sim = layout_similarity(tmp_path / "pages" / "page_0.png", tmp_path / "satisfaction" / "render_0.png")
    assert sim < MIN_LAYOUT_SIMILARITY, f"Expected low similarity for different images, got {sim:.2%}"


def test_layout_render_similarity_project_dir(tmp_path: Path) -> None:
    """
    For each page in project dir: require layout similarity between
    satisfaction/render_[n].png and pages/page_[n].png >= 95%; else fail with actual value.
    """
    if not _SSIM_AVAILABLE:
        pytest.skip("scikit-image not installed")
    (tmp_path / "pages").mkdir(parents=True)
    (tmp_path / "satisfaction").mkdir(parents=True)
    from PIL import Image
    import numpy as np
    # Two pages, both identical (render = page)
    for i in range(2):
        arr = np.ones((200, 320), dtype=np.uint8) * (180 + i * 20)
        img = Image.fromarray(arr, mode="L")
        img.save(tmp_path / "pages" / f"page_{i}.png")
        img.save(tmp_path / "satisfaction" / f"render_{i}.png")
    pairs = _collect_page_pairs(tmp_path)
    assert len(pairs) == 2
    for n, page_path, render_path in pairs:
        sim = layout_similarity(page_path, render_path)
        if sim < MIN_LAYOUT_SIMILARITY:
            pytest.fail(
                f"Page {n} layout similarity below threshold: required >={MIN_LAYOUT_SIMILARITY:.0%}, got {sim:.2%}. "
                f"Original: {page_path}, Render: {render_path}"
            )


def test_layout_render_similarity_skip_when_no_satisfaction(tmp_path: Path) -> None:
    """When project has pages/ but no satisfaction/, no pairs -> test can be skipped or pass."""
    if not _SSIM_AVAILABLE:
        pytest.skip("scikit-image not installed")
    (tmp_path / "pages").mkdir(parents=True)
    from PIL import Image
    import numpy as np
    Image.fromarray(np.ones((100, 100), dtype=np.uint8) * 255, mode="L").save(tmp_path / "pages" / "page_0.png")
    pairs = _collect_page_pairs(tmp_path)
    assert pairs == []


def test_layout_render_similarity_real_project_if_present() -> None:
    """
    If output/Testing has pages/ and satisfaction/render_*.png,
    require each page's layout similarity vs page_*.png >= 95%, else fail with actual value.
    """
    if not _SSIM_AVAILABLE:
        pytest.skip("scikit-image not installed")
    root = Path(__file__).resolve().parent.parent
    project_dir = root / "output" / "Testing"
    if not project_dir.is_dir():
        pytest.skip("output/Testing not found")
    pairs = _collect_page_pairs(project_dir)
    if not pairs:
        pytest.skip("output/Testing has no satisfaction/render_*.png to compare")
    failed: list[tuple[int, float]] = []
    for n, page_path, render_path in pairs:
        sim = layout_similarity(page_path, render_path)
        if sim < MIN_LAYOUT_SIMILARITY:
            failed.append((n, sim))
    if failed:
        details = "; ".join(f"page_{n}: {s:.2%}" for n, s in failed)
        pytest.fail(
            f"Layout similarity below threshold (required >={MIN_LAYOUT_SIMILARITY:.0%}). Actual: {details}"
        )
