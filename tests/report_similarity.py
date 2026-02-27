#!/usr/bin/env python3
"""
Report layout similarity: generated satisfaction/render_0.png vs render_after_modify.png and expected.png.
Outputs:
  - Pixel SSIM (sensitive to style)
  - Edge SSIM (structure only, more robust to hand-drawn vs typeset)
  - Position similarity (when expected.png has OCR cache; style-agnostic)
  - Gemini evaluation (when GOOGLE_API_KEY set): model rates layout similarity 0-100% and brief reason

Usage:
  python -m tests.report_similarity [project_dir]
  # or: python tests/report_similarity.py [project_dir]
  # Default project_dir: output/Testing
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import load_env, get_ocr_api_key
from src.layout.similarity import (
    layout_similarity,
    layout_similarity_edge,
    layout_similarity_position,
)


def _gemini_similarity(image_a: Path, image_b: Path, api_key: str) -> str | None:
    """Ask Gemini to rate layout similarity 0-100% and give brief reason. Returns response text or None."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None
    if not image_a.is_file() or not image_b.is_file():
        return None
    load_env()
    key = api_key or get_ocr_api_key()
    if not key:
        return None
    data_a = image_a.read_bytes()
    data_b = image_b.read_bytes()
    mime = "image/png"
    client = genai.Client(api_key=key)
    prompt = """You see two images: Image 1 is our generated layout render from the note; Image 2 is the reference (expected layout or original hand-drawn).
From a **layout structure** perspective, rate their similarity (0–100%) and give a one-sentence reason (e.g. whether block positions match, whether connector relationships correspond, whether overall arrangement is close). Ignore font, line thickness, and style; focus only on layout.
Output format exactly: Similarity: XX% Reason: …
Do not output anything else."""
    contents = [
        types.Part.from_bytes(data=data_a, mime_type=mime),
        types.Part.from_bytes(data=data_b, mime_type=mime),
        types.Part.from_text(text=prompt),
    ]
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=contents)
        raw = (response.text if response else "").strip()
        return raw if raw else None
    except Exception:
        return None


def _positions_from_ocr_json(path: Path) -> list[tuple[float, float]]:
    """Load (x_ratio, y_ratio) from OCR/refined JSON."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if isinstance(item, dict):
            x = item.get("x_ratio")
            y = item.get("y_ratio")
            if x is not None and y is not None:
                out.append((float(x), float(y)))
    return out


def main() -> int:
    project_dir = _ROOT / "output" / "Testing"
    if len(sys.argv) >= 2:
        project_dir = Path(sys.argv[1]).resolve()
    satisfaction = project_dir / "satisfaction"
    ocr_dir = project_dir / "ocr"
    render_0 = satisfaction / "render_0.png"
    if not render_0.is_file():
        print(f"Not found: {render_0}")
        print("Run first: python main.py --project Testing  (with playwright installed to generate render_0)")
        return 1
    load_env()
    common_portrait = (300, 400)
    print(f"Generated render: {render_0}")
    # render_after_modify
    after_path = satisfaction / "render_after_modify.png"
    if after_path.is_file():
        try:
            sim = layout_similarity(render_0, after_path, common_size=common_portrait)
            edge_sim = layout_similarity_edge(render_0, after_path, common_size=common_portrait)
            print(f"  vs render_after_modify.png:")
            print(f"    Pixel SSIM: {sim:.2%}  Edge/structure SSIM: {edge_sim:.2%}")
        except Exception as e:
            print(f"  render_after_modify.png: error - {e}")
    # expected
    expected_path = satisfaction / "expected.png"
    if not expected_path.is_file():
        expected_path = _ROOT / "sample" / "expected.png"
    if expected_path.is_file():
        try:
            sim = layout_similarity(render_0, expected_path, common_size=common_portrait)
            edge_sim = layout_similarity_edge(render_0, expected_path, common_size=common_portrait)
            print(f"  vs expected.png:")
            print(f"    Pixel SSIM: {sim:.2%}  Edge/structure SSIM: {edge_sim:.2%}")
        except Exception as e:
            print(f"  expected.png (pixel/edge): error - {e}")
        # Position similarity: our layout vs OCR positions from expected (optional; need ocr/expected.json)
        our_json = ocr_dir / "page_0_refined_ref.json"
        if not our_json.is_file():
            our_json = ocr_dir / "page_0.json"
        expected_ocr = ocr_dir / "expected.json"
        if our_json.is_file() and expected_ocr.is_file():
            try:
                pos_ours = _positions_from_ocr_json(our_json)
                pos_ref = _positions_from_ocr_json(expected_ocr)
                if pos_ours and pos_ref:
                    pos_sim = layout_similarity_position(pos_ours, pos_ref)
                    print(f"    Position similarity (layout only): {pos_sim:.2%}")
            except Exception as e:
                print(f"    Position similarity: error - {e}")
        # Gemini evaluation
        api_key = get_ocr_api_key()
        if api_key:
            gemini_result = _gemini_similarity(render_0, expected_path, api_key)
            if gemini_result:
                print(f"    Gemini evaluation: {gemini_result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
