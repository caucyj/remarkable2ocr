"""
Parse OCR lines into content-agnostic chart semantic JSON (ChartSchema) for the renderer.
Uses google-genai.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any

from .chart_schema import CHART_SCHEMA_INSTRUCTION

logger = logging.getLogger(__name__)


def semantic_parse(
    ocr_lines: list[dict[str, Any]],
    image_path: Path | str | None = None,
    *,
    api_key: str | None = None,
    model_name: str = "gemini-2.5-flash",
) -> dict[str, Any]:
    """
    Call LLM with OCR lines (+ optional image) and return ChartSchema JSON.
    """
    from ..config import get_ocr_api_key, load_env

    load_env()
    key = api_key or get_ocr_api_key()
    if not key:
        raise ValueError("Set GOOGLE_API_KEY or pass api_key")

    lines_text = []
    for i, row in enumerate(ocr_lines):
        t = row.get("text", "").strip()
        y = row.get("y_ratio", 0.5)
        x = row.get("x_ratio", 0.5)
        lines_text.append(f"[{i}] y={y:.2f} x={x:.2f} | {t}")
    ocr_block = "\n".join(lines_text)

    prompt = f"""Below is the OCR result of one page of handwritten notes. Each line format: [line index] y=vertical ratio x=horizontal ratio | recognized text.
Infer **structure** from **position and text** of these lines: outline hierarchy, boxes (rectangle/ellipse/longbar), arrow connections, list types, etc.
{CHART_SCHEMA_INSTRUCTION}

OCR lines (sorted by y):
{ocr_block}
"""

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)

    contents: list[Any] = []
    if image_path and Path(image_path).is_file():
        path = Path(image_path)
        data = path.read_bytes()
        contents.append(types.Part.from_bytes(data=data, mime_type="image/png" if path.suffix.lower() == ".png" else "image/jpeg"))
    contents.append(types.Part.from_text(text=prompt))

    logger.info("Gemini model (semantic chart): %s", model_name)
    logger.info("Semantic parse: calling Gemini API (chart structure)")
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
    )
    try:
        raw = response.text if response else ""
    except Exception as e:
        raise RuntimeError(f"LLM did not return text: {e}") from e
    raw = raw.strip()

    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {"outline": [], "containers": [], "arrows": [], "lists": []}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"outline": [], "containers": [], "arrows": [], "lists": []}

    return {
        "outline": obj.get("outline", []),
        "containers": obj.get("containers", []),
        "arrows": obj.get("arrows", []),
        "lists": obj.get("lists", []),
    }
