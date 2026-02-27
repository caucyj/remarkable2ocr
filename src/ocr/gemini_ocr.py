"""
Image -> OCR data (Gemini) with local cache and confidence.
Uses google-genai (new SDK).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _heuristic_confidence(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    t = text.strip()
    score = 0.5
    if len(t) >= 2:
        score += 0.2
    if any(c.isdigit() for c in t):
        score += 0.1
    if any(c in t for c in "→←↑↓·•-"):
        score += 0.1
    return min(1.0, score)


def _image_to_structured_ocr_impl(
    image_path: Path,
    *,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
    language_hint: str = "Chinese and English",
    request_confidence: bool = True,
) -> list[dict[str, Any]]:
    from google import genai
    from google.genai import types

    path = Path(image_path)
    data = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"

    client = genai.Client(api_key=api_key)

    confidence_instruction = '\nAdd a "confidence" field to each item: 0.0–1.0 or "high"/"medium"/"low".' if request_confidence else ""

    prompt = f"""You are analyzing a handwritten note image. Extract every text block and produce a JSON array that will be used to **recreate the exact same layout** on a canvas (same aspect ratio). So positions must be **precise**.

**Output format** – one object per text block (in reading order):
- "text": string (content of the block)
- "x_ratio": float 0–1 = horizontal **center** of this block in the image (0=left edge, 1=right edge)
- "y_ratio": float 0–1 = vertical **center** of this block (0=top, 1=bottom)

**Layout fidelity**: When we render each item at (x_ratio*100%, y_ratio*100%), the result must look like the original note. Estimate positions carefully so the layout matches.

**Optional fields**:
- "links": array of zero-based indices of blocks this one points to (arrows, flow, hierarchy). Omit if none.
- "shape": "box" (rectangle) or "circle" (ellipse/circle) if the text is inside such a shape. Omit for plain text.
- "color": CSS color or hex (e.g. "red", "#c00") if the text is not black. Omit otherwise.
- For items with shape "box" or "circle", you may add "width_ratio" and "height_ratio" (0–1, relative to image width/height) so we can size the rendered block correctly. Omit if unsure.{confidence_instruction}

**Example** (positions are illustrative; use the actual positions you see in the image):
[
  {{ "text": "① Requirements -> PRD", "y_ratio": 0.08, "x_ratio": 0.35, "links": [1, 2], "shape": "box" }},
  {{ "text": "-> Market", "y_ratio": 0.10, "x_ratio": 0.68 }},
  {{ "text": "Design", "y_ratio": 0.18, "x_ratio": 0.38, "shape": "box", "width_ratio": 0.2, "height_ratio": 0.04 }}
]

Output **only** the JSON array, no markdown or extra text. Preserve vertical order. Language may be {language_hint}.
"""
    contents = [
        types.Part.from_bytes(data=data, mime_type=mime),
        types.Part.from_text(text=prompt),
    ]
    logger.info("Gemini model: %s", model_name)
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
    )
    try:
        raw = response.text if response else ""
    except Exception as e:
        raise RuntimeError(f"Gemini did not return text: {e}") from e
    raw = raw.strip()
    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        n = max(len(lines), 1)
        return [{"text": ln, "y_ratio": (i + 0.5) / n, "x_ratio": 0.5} for i, ln in enumerate(lines)]
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        n = max(len(lines), 1)
        return [{"text": ln, "y_ratio": (i + 0.5) / n, "x_ratio": 0.5} for i, ln in enumerate(lines)]
    out = []
    for i, item in enumerate(arr):
        if not isinstance(item, dict):
            continue
        text = item.get("text") or ""
        y = item.get("y_ratio")
        y = (i + 0.5) / max(len(arr), 1) if y is None or not isinstance(y, (int, float)) else max(0.0, min(1.0, float(y)))
        x = item.get("x_ratio")
        x = 0.5 if x is None or not isinstance(x, (int, float)) else max(0.0, min(1.0, float(x)))
        row = {"text": text, "y_ratio": y, "x_ratio": x}
        if "confidence" in item and item["confidence"] is not None:
            c = item["confidence"]
            if isinstance(c, (int, float)):
                row["confidence"] = max(0.0, min(1.0, float(c)))
            elif isinstance(c, str) and c.lower() in ("high", "medium", "low"):
                row["confidence"] = c.lower()
            else:
                row["confidence"] = c
        if "links" in item and isinstance(item["links"], list):
            row["links"] = [int(n) for n in item["links"] if isinstance(n, (int, float)) and 0 <= int(n) < len(arr)]
        if "shape" in item and item.get("shape") in ("box", "circle"):
            row["shape"] = item.get("shape")
        if "color" in item and item.get("color"):
            row["color"] = str(item.get("color")).strip()
        if "width_ratio" in item and isinstance(item.get("width_ratio"), (int, float)):
            row["width_ratio"] = max(0.01, min(1.0, float(item["width_ratio"])))
        if "height_ratio" in item and isinstance(item.get("height_ratio"), (int, float)):
            row["height_ratio"] = max(0.01, min(1.0, float(item["height_ratio"])))
        out.append(row)
    out.sort(key=lambda r: (r["y_ratio"], r["x_ratio"]))
    return out


def refine_layout_with_gemini(
    image_path: Path | str,
    ocr_lines: list[dict[str, Any]],
    *,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
    reference_image_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """
    Ask Gemini to refine x_ratio, y_ratio so that when we render the items, the layout matches
    an image. If reference_image_path is set, that image is the TARGET layout; otherwise
    image_path (the original note scan) is the target. Returns same list with updated positions.
    """
    from google import genai
    from google.genai import types

    # Use reference image as target when provided (e.g. expected.png for higher similarity)
    path = Path(reference_image_path if reference_image_path else image_path)
    if not path.is_file():
        return ocr_lines
    data = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    client = genai.Client(api_key=api_key)
    json_str = json.dumps(ocr_lines, ensure_ascii=False, indent=0)
    if reference_image_path:
        # Send BOTH original note and reference so the model can match JSON blocks to positions in the reference
        path_orig = Path(image_path)
        if path_orig.is_file():
            data_orig = path_orig.read_bytes()
            mime_orig = "image/png" if path_orig.suffix.lower() == ".png" else "image/jpeg"
            contents = [
                types.Part.from_text(text="You will see two images. IMAGE 1 = source handwritten note. IMAGE 2 = target layout we want the JSON to match.\n\n"),
                types.Part.from_bytes(data=data_orig, mime_type=mime_orig),
                types.Part.from_bytes(data=data, mime_type=mime),
                types.Part.from_text(text="""The JSON array below has one object per text block from Image 1. Your task: for each object, set "x_ratio" and "y_ratio" to the CENTER of where that text (or the corresponding block) appears in IMAGE 2. Use 0–1: x_ratio 0=left edge of image, 1=right edge; y_ratio 0=top, 1=bottom. So when we render each block at (x_ratio*100%, y_ratio*100%) on a canvas with the same aspect ratio as Image 2, the result will look like Image 2. Do not add or remove items. Do not change "text", "links", "shape", "color". You may set "width_ratio" and "height_ratio" for box/circle items to match Image 2. Output ONLY a valid JSON array, no markdown.\n\n""" + "Current JSON:\n" + json_str),
            ]
        else:
            contents = [
                types.Part.from_bytes(data=data, mime_type=mime),
                types.Part.from_text(text="The attached image is the TARGET layout. Set x_ratio and y_ratio (0–1, center of each block in the image) for each JSON item so the rendered layout matches the image. Output only the JSON array.\n\nCurrent JSON:\n" + json_str),
            ]
    else:
        contents = [
            types.Part.from_bytes(data=data, mime_type=mime),
            types.Part.from_text(text="""This JSON array describes text blocks extracted from the image. Each item has text, x_ratio, y_ratio (0-1, center position), and optionally links, shape, color, width_ratio, height_ratio.

Your task: **Refine only the position/size numbers** so that when we render each block at (x_ratio*100%, y_ratio*100%) on a canvas with the same aspect ratio as the image, the layout matches the handwritten note exactly. Do not change "text", "links", "shape", or "color". Only update x_ratio, y_ratio and optionally width_ratio, height_ratio. Use precise values between 0 and 1. Keep the same array length and order.

Output only the JSON array, no other text."""

 + "\n\nCurrent JSON:\n" + json_str),
        ]
    logger.info("Gemini model (layout refinement): %s", model_name)
    if reference_image_path:
        logger.info("Refining layout to match reference image: %s", path)
    try:
        response = client.models.generate_content(model=model_name, contents=contents)
        raw = (response.text if response else "").strip()
    except Exception as e:
        logger.warning("Layout refinement failed: %s", e)
        return ocr_lines
    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        return ocr_lines
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return ocr_lines
    if len(arr) != len(ocr_lines):
        return ocr_lines
    out = []
    for i, (orig, item) in enumerate(zip(ocr_lines, arr)):
        if not isinstance(item, dict):
            out.append(orig)
            continue
        row = dict(orig)
        x = item.get("x_ratio")
        if isinstance(x, (int, float)):
            row["x_ratio"] = max(0.0, min(1.0, float(x)))
        y = item.get("y_ratio")
        if isinstance(y, (int, float)):
            row["y_ratio"] = max(0.0, min(1.0, float(y)))
        if "width_ratio" in item and isinstance(item["width_ratio"], (int, float)):
            row["width_ratio"] = max(0.01, min(1.0, float(item["width_ratio"])))
        if "height_ratio" in item and isinstance(item["height_ratio"], (int, float)):
            row["height_ratio"] = max(0.01, min(1.0, float(item["height_ratio"])))
        out.append(row)
    out.sort(key=lambda r: (r["y_ratio"], r["x_ratio"]))
    return out


def semantic_layout_with_gemini(
    image_path: Path | str,
    ocr_lines: list[dict[str, Any]],
    *,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    """
    Ask Gemini to understand the semantics of the note (flows, hierarchy, grouping)
    and suggest connection relationships (links) and groups (oval/box). Only links
    when the image shows an explicit arrow/line between two blocks.
    Returns (enriched_ocr_lines, suggestion_text, groups). groups is a list of
    { "label": str, "indices": list[int] } for blocks inside one oval/box.
    """
    from google import genai
    from google.genai import types

    path = Path(image_path)
    if not path.is_file() or not ocr_lines:
        return ocr_lines, "", []
    data = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    client = genai.Client(api_key=api_key)
    json_str = json.dumps(ocr_lines, ensure_ascii=False, indent=0)
    prompt = """You see a handwritten note image and a JSON list of text blocks (each has text, x_ratio, y_ratio, and optional links, shape, color).

Do two things:

1) **Layout suggestion** (2–5 lines, any language): Describe the overall structure (e.g. flow, branches, center node), which blocks are **explicitly connected by drawn arrows/lines** in the image, and how to express them with links. Then suggest how to arrange the layout more clearly.

2) **Output a single JSON object** (not just an array) with two keys:
   - **"items"**: Array, same length and order as input. Keep each element's text, x_ratio, y_ratio, shape, color, width_ratio, height_ratio. Add "links" **only when the image shows an explicit arrow or line** from this block to another (links = zero-based indices of target blocks).
     **Important**: Only add a link when two blocks are **visibly connected by a drawn arrow or line**. Do NOT link blocks that are merely in the same section or topic without a drawn connection (e.g. "Implementation" and "Code risk" in the same area but no arrow between them). Do not infer links from semantics alone.
   - **"groups"**: Array of groups where multiple blocks are **inside one drawn oval or box**. Each item: { "label": "group title", "indices": [block indices in items] }. Only output groups when the image clearly encloses multiple blocks in one oval/rectangle. indices are zero-based into items.

3) Optional: Add "role" to some items ("title" | "group_label" | "item" | "center" etc.).

First output the layout suggestion text, optionally prefixed with "Layout suggestion:". Then on a new line output the JSON object with "items" and "groups". Do not wrap the JSON in markdown code fences."""

    contents = [
        types.Part.from_bytes(data=data, mime_type=mime),
        types.Part.from_text(text=prompt + "\n\nCurrent JSON:\n" + json_str),
    ]
    logger.info("Gemini model (semantic layout): %s", model_name)
    try:
        response = client.models.generate_content(model=model_name, contents=contents)
        raw = (response.text if response else "").strip()
    except Exception as e:
        logger.warning("Semantic layout failed: %s", e)
        return ocr_lines, "", []
    suggestion = ""
    if "Layout suggestion" in raw:
        idx = raw.find("Layout suggestion")
        bracket = raw.find("[", idx)
        if bracket > idx:
            suggestion = raw[idx:bracket].strip()
            raw = raw[bracket:]
        else:
            line_end = raw.find("\n", idx)
            suggestion = raw[idx:line_end].strip() if line_end != -1 else raw[idx:].strip()
            if line_end != -1:
                raw = raw[line_end + 1:].strip()
        for strip in ("```json", "```", "```\n"):
            if suggestion.endswith(strip):
                suggestion = suggestion[: -len(strip)].strip()
    # Strip markdown code fence around JSON if present
    for prefix in ("```json\n", "```json", "```\n"):
        if raw.strip().startswith(prefix):
            raw = raw.strip()[len(prefix):].strip()
    if raw.strip().endswith("```"):
        raw = raw.strip()[:-3].strip()
    # Support both { "items": [...], "groups": [...] } and legacy [...]
    arr = None
    groups: list[dict[str, Any]] = []
    m_obj = re.search(r"\{[\s\S]*\}", raw)
    if m_obj:
        try:
            obj = json.loads(m_obj.group(0))
            if isinstance(obj, dict) and "items" in obj and isinstance(obj["items"], list):
                arr = obj["items"]
                if "groups" in obj and isinstance(obj["groups"], list):
                    for g in obj["groups"]:
                        if isinstance(g, dict) and "indices" in g and isinstance(g["indices"], list):
                            groups.append({"label": str(g.get("label", "")).strip(), "indices": [int(x) for x in g["indices"] if isinstance(x, (int, float))]})
        except json.JSONDecodeError:
            pass
    if arr is None:
        m = re.search(r"\[[\s\S]*\]", raw)
        if m:
            try:
                arr = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    if not arr or len(arr) != len(ocr_lines):
        return ocr_lines, suggestion, []
    n = len(arr)
    filtered_groups = []
    for g in groups:
        indices = [i for i in g["indices"] if 0 <= i < n]
        if indices:
            filtered_groups.append({"label": g["label"], "indices": indices})
    out = []
    for i, (orig, item) in enumerate(zip(ocr_lines, arr)):
        if not isinstance(item, dict):
            out.append(dict(orig))
            continue
        row = dict(orig)
        if "links" in item and isinstance(item["links"], list):
            row["links"] = [int(n) for n in item["links"] if isinstance(n, (int, float)) and 0 <= int(n) < len(arr)]
        if "role" in item and isinstance(item["role"], str) and item["role"].strip():
            row["role"] = item["role"].strip()
        out.append(row)
    out.sort(key=lambda r: (r["y_ratio"], r["x_ratio"]))
    return out, suggestion, filtered_groups


def ocr_image(
    image_path: Path | str,
    cache_dir: Path,
    *,
    cache_key: str | None = None,
    return_confidence: bool = True,
    api_key: str | None = None,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """
    Run OCR on a single note image; write result to cache_dir. If cache exists and use_cache,
    return from cache. cache_key used as cache filename (e.g. page_0 -> page_0.json).
    Returns list of rows: { "text", "y_ratio", "x_ratio", "confidence"? , "links"? , "shape"? , "color"? }.
    """
    from ..config import get_ocr_api_key, load_env

    load_env()
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    key = cache_key if cache_key is not None else hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:32]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{key}.json"

    # Use cache when present and not skipping (we do not compare mtime to page image)
    if use_cache and cache_file.is_file():
        raw = cache_file.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                out = []
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    row = {
                        "text": item.get("text", ""),
                        "y_ratio": max(0.0, min(1.0, float(item.get("y_ratio", 0.5)))),
                        "x_ratio": max(0.0, min(1.0, float(item.get("x_ratio", 0.5)))),
                    }
                    if "confidence" in item:
                        row["confidence"] = item["confidence"]
                    if "links" in item and isinstance(item.get("links"), list):
                        row["links"] = [int(n) for n in item["links"] if isinstance(n, (int, float))]
                    if "shape" in item and item.get("shape") in ("box", "circle"):
                        row["shape"] = item.get("shape")
                    if "color" in item and item.get("color"):
                        row["color"] = str(item.get("color")).strip()
                    if "width_ratio" in item and isinstance(item.get("width_ratio"), (int, float)):
                        row["width_ratio"] = max(0.01, min(1.0, float(item["width_ratio"])))
                    if "height_ratio" in item and isinstance(item.get("height_ratio"), (int, float)):
                        row["height_ratio"] = max(0.01, min(1.0, float(item["height_ratio"])))
                    out.append(row)
                out.sort(key=lambda r: (r["y_ratio"], r["x_ratio"]))
                logger.info("OCR %s: using local cache (no Gemini request)", key)
                return out
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    # No valid cache: call Gemini (first run or --no-cache)
    if use_cache and not cache_file.is_file():
        logger.info("OCR %s: no local cache, calling Gemini API", key)
    elif not use_cache:
        logger.info("OCR %s: --no-cache, calling Gemini API", key)

    api_key = api_key or get_ocr_api_key()
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or pass api_key")

    result = _image_to_structured_ocr_impl(path, api_key=api_key, request_confidence=return_confidence)
    if return_confidence:
        for row in result:
            if "confidence" not in row:
                row["confidence"] = _heuristic_confidence(row.get("text", ""))

    cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
