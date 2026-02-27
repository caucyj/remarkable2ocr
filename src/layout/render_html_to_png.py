"""
Render layout.html to satisfaction/render_[n].png via headless browser (Playwright).
Each .ocr-page-wrap in the HTML is screenshot to render_0.png, render_1.png, ...
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from playwright.sync_api import sync_playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False


def render_layout_to_satisfaction(
    layout_html_path: Path,
    satisfaction_dir: Path,
    *,
    viewport_width: int = 800,
    viewport_height: int = 1000,
) -> list[Path]:
    """
    Open layout.html and screenshot each .ocr-page-wrap to satisfaction/render_0.png, render_1.png, ...
    Returns list of written paths, or [] if Playwright not available or no pages found.
    """
    if not _PLAYWRIGHT_AVAILABLE:
        logger.warning(
            "Playwright not installed; skipping layout -> satisfaction render. "
            "Install with: pip install playwright && playwright install chromium"
        )
        return []
    layout_html_path = Path(layout_html_path).resolve()
    if not layout_html_path.is_file():
        logger.warning("Layout HTML not found: %s", layout_html_path)
        return []
    satisfaction_dir = Path(satisfaction_dir).resolve()
    satisfaction_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": viewport_width, "height": viewport_height})
                url = layout_html_path.as_uri()
                page.goto(url, wait_until="networkidle", timeout=15000)
                page.wait_for_timeout(500)
                wraps = page.locator(".ocr-page-wrap")
                n_wraps = wraps.count()
                for i in range(n_wraps):
                    out_path = satisfaction_dir / f"render_{i}.png"
                    wraps.nth(i).screenshot(path=str(out_path), type="png")
                    written.append(out_path)
                    logger.info("  satisfaction: %s", out_path.name)
            finally:
                browser.close()
    except Exception as e:
        logger.warning("Layout render to PNG failed: %s", e)
        return []
    return written
