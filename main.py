#!/usr/bin/env python3
"""
Root entry: scan data/xochitl notebooks -> render -> OCR -> layout.
Supports --pull (rsync from reMarkable) and --camera (OCR a single image).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import load_env, get_data_dir, get_ocr_api_key
from src.remarkable import list_notebooks, render_notebook_pages, pull_xochitl
from src.ocr import ocr_image, refine_layout_with_gemini, semantic_layout_with_gemini
from src.layout import write_ocr_preview_html, render_ocr_overlay, render_ocr_to_html_multi, build_xmind
from src.layout.render_html_to_png import render_layout_to_satisfaction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _safe_notebook_name(name: str) -> str:
    """Turn notebook visible name into a filesystem-safe directory name."""
    s = re.sub(r'[/\\:*?"<>|]', "", name)
    s = s.strip() or "unnamed"
    s = re.sub(r"\s+", "_", s)
    return s[:200]


def _log_project_not_found(project_name: str, output_root: Path, data_dir: Path) -> None:
    """Emit a friendly error when --project name is not found and there is no output cache."""
    out_dir = output_root / project_name
    has_cache = out_dir.is_dir() and any((out_dir / "ocr").glob("*.json"))
    logger.error(
        "Project '%s' not found. No notebook in data/xochitl has (safe) name '%s'.",
        project_name, project_name,
    )
    if not has_cache:
        logger.error(
            "No existing OCR cache at output/%s/. List notebooks in data/xochitl or use --camera %s for a camera project.",
            project_name, project_name,
        )
    else:
        logger.error("You have cached output at output/%s/ but no matching notebook in data/xochitl.", project_name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan data/xochitl notebooks -> render -> OCR -> layout. Supports --pull and --camera."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore local OCR cache and call Gemini for every page",
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull xochitl from reMarkable via SSH+rsync before processing; fail if not connected",
    )
    parser.add_argument(
        "--camera",
        metavar="PROJECT_NAME",
        default=None,
        help="Run OCR on a single image in data/xochitl/camera/PROJECT_NAME/ and output to output/PROJECT_NAME/",
    )
    parser.add_argument(
        "--xmind",
        action="store_true",
        help="After generating layout.html, also export OCR structure to project_name.xmind (mind map)",
    )
    parser.add_argument(
        "--project",
        metavar="PROJECT_NAME",
        default=None,
        help="Process only this project (notebook with matching safe name). Fails with a friendly message if not found and no cache at output/PROJECT_NAME/.",
    )
    parser.add_argument(
        "--refine-layout",
        action="store_true",
        help="After OCR, call Gemini to refine x_ratio/y_ratio so rendered layout matches the note image (or --refine-to image). Cache: page_N_refined.json or page_N_refined_ref.json",
    )
    parser.add_argument(
        "--refine-to",
        metavar="IMAGE_PATH",
        default=None,
        help="When used with --refine-layout: use this image as the target layout (e.g. output/Testing/satisfaction/expected.png). If omitted, use project satisfaction/expected.png when present.",
    )
    parser.add_argument(
        "--semantic-layout",
        action="store_true",
        help="After OCR, call Gemini to understand semantics (flows, hierarchy, grouping) and suggest links + layout. Logs layout suggestion; cache: page_N_semantic.json",
    )
    args = parser.parse_args()
    use_ocr_cache = not args.no_cache

    load_env()
    data_dir = get_data_dir()

    if args.pull:
        try:
            pull_xochitl(data_dir)
        except Exception as e:
            logger.error("%s", e)
            return 1

    if not data_dir.is_dir():
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created data directory: %s", data_dir)

    output_root = _ROOT / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    logger.info("Data dir: %s, output root: %s", data_dir, output_root)

    if args.camera is not None:
        return _run_camera_mode(data_dir, output_root, args.camera, use_ocr_cache, args.xmind, args.refine_layout, args.refine_to, args.semantic_layout)

    return _run_notebook_mode(data_dir, output_root, use_ocr_cache, args.xmind, args.project, args.refine_layout, args.refine_to, args.semantic_layout)


def _run_camera_mode(
    data_dir: Path,
    output_root: Path,
    project_name: str,
    use_ocr_cache: bool,
    use_xmind: bool = False,
    refine_layout: bool = False,
    refine_to: str | None = None,
    semantic_layout: bool = False,
) -> int:
    """Process all images in data/xochitl/camera/<project_name>/."""
    camera_dir = data_dir / "camera" / project_name
    if not camera_dir.is_dir():
        camera_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created camera project directory: %s", camera_dir)
    images = sorted(
        p for p in camera_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        logger.error(
            "No image found in %s (supported: %s). Add a .png or .jpg file and run again.",
            camera_dir, ", ".join(IMAGE_EXTENSIONS),
        )
        return 1
    out_dir = output_root / project_name
    pages_dir = out_dir / "pages"
    ocr_dir = out_dir / "ocr"
    pages_dir.mkdir(parents=True, exist_ok=True)
    ocr_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Camera project: %s -> %s (%d image(s))", project_name, out_dir, len(images))
    t0 = time.perf_counter()
    page_paths: list[Path] = []
    for i, image_path in enumerate(images):
        page_path = pages_dir / f"page_{i}.png"
        if image_path.suffix.lower() == ".png":
            shutil.copy2(image_path, page_path)
        else:
            try:
                from PIL import Image
                Image.open(image_path).convert("RGB").save(page_path, "PNG")
            except Exception:
                shutil.copy2(image_path, page_path)
        page_paths.append(page_path)
    all_ocr: list[list[dict]] = []
    for i, p_path in enumerate(page_paths):
        t_page = time.perf_counter()
        try:
            ocr_lines = ocr_image(
                p_path,
                ocr_dir,
                cache_key=f"page_{i}",
                return_confidence=True,
                use_cache=use_ocr_cache,
            )
            all_ocr.append(ocr_lines)
        except Exception as e:
            logger.warning("  OCR page_%d failed: %s", i, e)
            cache_file = ocr_dir / f"page_{i}.json"
            if cache_file.is_file():
                try:
                    all_ocr.append(json.loads(cache_file.read_text(encoding="utf-8")))
                except Exception:
                    all_ocr.append([])
            else:
                all_ocr.append([])
        logger.info("  Page %d/%d OCR done in %.2fs", i + 1, len(page_paths), time.perf_counter() - t_page)
    debug_dir = out_dir / ".debug"
    if all_ocr:
        debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            write_ocr_preview_html(all_ocr, debug_dir / "ocr_preview.html")
            logger.info("  .debug: ocr_preview.html")
        except Exception as e:
            logger.warning("  Failed to write ocr_preview.html: %s", e)
        for i, (ocr_lines, p_path) in enumerate(zip(all_ocr, page_paths)):
            if not ocr_lines:
                continue
            try:
                render_ocr_overlay(ocr_lines, p_path, debug_dir / f"ocr_overlay_{i}.png")
                logger.info("  .debug: ocr_overlay_%d.png", i)
            except Exception as e:
                logger.warning("  Failed to write ocr_overlay_%d.png: %s", i, e)
    if not all_ocr:
        logger.warning("No OCR result, skipping layout")
        return 0
    if semantic_layout:
        api_key = get_ocr_api_key()
        all_groups: list[list[dict]] = [[] for _ in range(len(all_ocr))]
        if api_key:
            for i, (p_path, ocr_lines) in enumerate(zip(page_paths, all_ocr)):
                if not ocr_lines:
                    continue
                cache_semantic = ocr_dir / f"page_{i}_semantic.json"
                if use_ocr_cache and cache_semantic.is_file():
                    try:
                        data = json.loads(cache_semantic.read_text(encoding="utf-8"))
                        if isinstance(data, list):
                            all_ocr[i] = data
                            all_groups[i] = []
                            logger.info("  Semantic layout page_%d: using cache", i)
                        elif isinstance(data, dict) and "items" in data:
                            all_ocr[i] = data["items"]
                            all_groups[i] = data.get("groups") if isinstance(data.get("groups"), list) else []
                            logger.info("  Semantic layout page_%d: using cache", i)
                        else:
                            raise ValueError("invalid cache")
                    except Exception:
                        enriched, suggestion, groups = semantic_layout_with_gemini(p_path, ocr_lines, api_key=api_key)
                        all_ocr[i] = enriched
                        all_groups[i] = groups
                        cache_semantic.write_text(json.dumps({"items": enriched, "groups": groups}, ensure_ascii=False, indent=2), encoding="utf-8")
                        if suggestion:
                            logger.info("  Layout suggestion page_%d: %s", i, suggestion)
                        logger.info("  Semantic layout page_%d: Gemini", i)
                else:
                    enriched, suggestion, groups = semantic_layout_with_gemini(p_path, ocr_lines, api_key=api_key)
                    all_ocr[i] = enriched
                    all_groups[i] = groups
                    cache_semantic.write_text(json.dumps({"items": enriched, "groups": groups}, ensure_ascii=False, indent=2), encoding="utf-8")
                    if suggestion:
                        logger.info("  Layout suggestion page_%d: %s", i, suggestion)
                    logger.info("  Semantic layout page_%d: Gemini", i)
        else:
            logger.warning("  --semantic-layout skipped: no GOOGLE_API_KEY")
            all_groups = [[] for _ in range(len(all_ocr))]
    if refine_layout:
        api_key = get_ocr_api_key()
        if api_key:
            ref_path = Path(refine_to) if refine_to else (out_dir / "satisfaction" / "expected.png")
            ref_path = ref_path.resolve() if ref_path.is_file() else None
            cache_suffix = "_ref" if ref_path else ""
            for i, (p_path, ocr_lines) in enumerate(zip(page_paths, all_ocr)):
                if not ocr_lines:
                    continue
                cache_refined = ocr_dir / f"page_{i}_refined{cache_suffix}.json"
                if use_ocr_cache and cache_refined.is_file():
                    try:
                        all_ocr[i] = json.loads(cache_refined.read_text(encoding="utf-8"))
                        logger.info("  Layout refinement page_%d: using cache", i)
                    except Exception:
                        refined = refine_layout_with_gemini(p_path, ocr_lines, api_key=api_key, reference_image_path=ref_path)
                        all_ocr[i] = refined
                        cache_refined.write_text(json.dumps(refined, ensure_ascii=False, indent=2), encoding="utf-8")
                        logger.info("  Layout refinement page_%d: Gemini", i)
                else:
                    refined = refine_layout_with_gemini(p_path, ocr_lines, api_key=api_key, reference_image_path=ref_path)
                    all_ocr[i] = refined
                    cache_refined.write_text(json.dumps(refined, ensure_ascii=False, indent=2), encoding="utf-8")
                    logger.info("  Layout refinement page_%d: Gemini", i)
        else:
            logger.warning("  --refine-layout skipped: no GOOGLE_API_KEY")
    try:
        page_groups = all_groups if semantic_layout else None
        render_ocr_to_html_multi(all_ocr, out_dir / "layout.html", use_raw_ratios=refine_layout, page_groups=page_groups)
        logger.info("Layout: layout.html (%d page(s))", len(all_ocr))
        satisfaction_dir = out_dir / "satisfaction"
        rendered = render_layout_to_satisfaction(out_dir / "layout.html", satisfaction_dir)
        if rendered:
            logger.info("Rendered %d page(s) to %s", len(rendered), satisfaction_dir)
    except Exception as e:
        logger.warning("Layout failed: %s", e)
    if use_xmind:
        try:
            xmind_path = build_xmind(all_ocr, out_dir / f"{project_name}.xmind", sheet_title=project_name)
            logger.info("XMind: %s", xmind_path.name)
        except Exception as e:
            logger.warning("XMind export failed: %s", e)
    logger.info("Camera project %s completed in %.2fs", project_name, time.perf_counter() - t0)
    logger.info("Output: %s", output_root)
    return 0


def _run_notebook_mode(
    data_dir: Path,
    output_root: Path,
    use_ocr_cache: bool,
    use_xmind: bool = False,
    project_name_filter: str | None = None,
    refine_layout: bool = False,
    refine_to: str | None = None,
    semantic_layout: bool = False,
) -> int:
    """Scan notebooks and process each (or only project_name_filter if set)."""
    notebooks = list_notebooks(data_dir)
    if not notebooks:
        if project_name_filter:
            _log_project_not_found(project_name_filter, output_root, data_dir)
            return 1
        logger.warning("No notebooks found")
        return 0

    if project_name_filter is not None:
        matching = [nb for nb in notebooks if _safe_notebook_name(nb.visible_name) == project_name_filter]
        if not matching:
            _log_project_not_found(project_name_filter, output_root, data_dir)
            return 1
        notebooks = matching

    total_notebooks = len(notebooks)
    for idx, nb in enumerate(notebooks):
        nb_start = time.perf_counter()
        safe_name = _safe_notebook_name(nb.visible_name)
        out_dir = output_root / safe_name
        pages_dir = out_dir / "pages"
        ocr_dir = out_dir / "ocr"
        pages_dir.mkdir(parents=True, exist_ok=True)
        ocr_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Notebook %d/%d: %s -> %s", idx + 1, total_notebooks, nb.visible_name, out_dir)

        page_paths = render_notebook_pages(nb, pages_dir)
        if not page_paths:
            logger.warning("  No pages rendered, skipping OCR and layout")
            continue
        logger.info("  Pages: %d -> %s", len(page_paths), pages_dir)

        all_ocr: list[list[dict]] = []
        for i, p_path in enumerate(page_paths):
            t0 = time.perf_counter()
            try:
                ocr_lines = ocr_image(
                    p_path,
                    ocr_dir,
                    cache_key=f"page_{i}",
                    return_confidence=True,
                    use_cache=use_ocr_cache,
                )
                all_ocr.append(ocr_lines)
            except Exception as e:
                logger.warning("  OCR page_%d failed: %s", i, e)
                cache_file = ocr_dir / f"page_{i}.json"
                if cache_file.is_file():
                    try:
                        raw = cache_file.read_text(encoding="utf-8")
                        all_ocr.append(json.loads(raw))
                        logger.info("  Using cache: %s (%.2fs)", cache_file.name, time.perf_counter() - t0)
                    except Exception:
                        all_ocr.append([])
                else:
                    all_ocr.append([])
            else:
                logger.info("  Page %d/%d OCR done in %.2fs", i + 1, len(page_paths), time.perf_counter() - t0)

        debug_dir = out_dir / ".debug"
        if all_ocr:
            debug_dir.mkdir(parents=True, exist_ok=True)
            try:
                write_ocr_preview_html(all_ocr, debug_dir / "ocr_preview.html")
                logger.info("  .debug: ocr_preview.html")
            except Exception as e:
                logger.warning("  Failed to write ocr_preview.html: %s", e)
            for i, (ocr_lines, p_path) in enumerate(zip(all_ocr, page_paths)):
                if not ocr_lines:
                    continue
                try:
                    render_ocr_overlay(
                        ocr_lines,
                        p_path,
                        debug_dir / f"ocr_overlay_{i}.png",
                    )
                    logger.info("  .debug: ocr_overlay_%d.png", i)
                except Exception as e:
                    logger.warning("  Failed to write ocr_overlay_%d.png: %s", i, e)

        if not all_ocr:
            logger.warning("  No OCR result, skipping layout")
            continue
        if semantic_layout:
            api_key = get_ocr_api_key()
            all_groups = [[] for _ in range(len(all_ocr))]
            if api_key:
                for i, (p_path, ocr_lines) in enumerate(zip(page_paths, all_ocr)):
                    if not ocr_lines:
                        continue
                    cache_semantic = ocr_dir / f"page_{i}_semantic.json"
                    if use_ocr_cache and cache_semantic.is_file():
                        try:
                            data = json.loads(cache_semantic.read_text(encoding="utf-8"))
                            if isinstance(data, list):
                                all_ocr[i] = data
                                all_groups[i] = []
                                logger.info("  Semantic layout page_%d: using cache", i)
                            elif isinstance(data, dict) and "items" in data:
                                all_ocr[i] = data["items"]
                                all_groups[i] = data.get("groups") if isinstance(data.get("groups"), list) else []
                                logger.info("  Semantic layout page_%d: using cache", i)
                            else:
                                raise ValueError("invalid cache")
                        except Exception:
                            enriched, suggestion, groups = semantic_layout_with_gemini(p_path, ocr_lines, api_key=api_key)
                            all_ocr[i] = enriched
                            all_groups[i] = groups
                            cache_semantic.write_text(json.dumps({"items": enriched, "groups": groups}, ensure_ascii=False, indent=2), encoding="utf-8")
                            if suggestion:
                                logger.info("  Layout suggestion page_%d: %s", i, suggestion)
                            logger.info("  Semantic layout page_%d: Gemini", i)
                    else:
                        enriched, suggestion, groups = semantic_layout_with_gemini(p_path, ocr_lines, api_key=api_key)
                        all_ocr[i] = enriched
                        all_groups[i] = groups
                        cache_semantic.write_text(json.dumps({"items": enriched, "groups": groups}, ensure_ascii=False, indent=2), encoding="utf-8")
                        if suggestion:
                            logger.info("  Layout suggestion page_%d: %s", i, suggestion)
                        logger.info("  Semantic layout page_%d: Gemini", i)
            else:
                logger.warning("  --semantic-layout skipped: no GOOGLE_API_KEY")
        else:
            all_groups = [[] for _ in range(len(all_ocr))]
        if refine_layout:
            api_key = get_ocr_api_key()
            if api_key:
                ref_path = Path(refine_to) if refine_to else (out_dir / "satisfaction" / "expected.png")
                ref_path = ref_path.resolve() if ref_path.is_file() else None
                cache_suffix = "_ref" if ref_path else ""
                for i, (p_path, ocr_lines) in enumerate(zip(page_paths, all_ocr)):
                    if not ocr_lines:
                        continue
                    cache_refined = ocr_dir / f"page_{i}_refined{cache_suffix}.json"
                    if use_ocr_cache and cache_refined.is_file():
                        try:
                            all_ocr[i] = json.loads(cache_refined.read_text(encoding="utf-8"))
                            logger.info("  Layout refinement page_%d: using cache", i)
                        except Exception:
                            refined = refine_layout_with_gemini(p_path, ocr_lines, api_key=api_key, reference_image_path=ref_path)
                            all_ocr[i] = refined
                            cache_refined.write_text(json.dumps(refined, ensure_ascii=False, indent=2), encoding="utf-8")
                            logger.info("  Layout refinement page_%d: Gemini", i)
                    else:
                        refined = refine_layout_with_gemini(p_path, ocr_lines, api_key=api_key, reference_image_path=ref_path)
                        all_ocr[i] = refined
                        cache_refined.write_text(json.dumps(refined, ensure_ascii=False, indent=2), encoding="utf-8")
                        logger.info("  Layout refinement page_%d: Gemini", i)
            else:
                logger.warning("  --refine-layout skipped: no GOOGLE_API_KEY")
        try:
            page_groups = all_groups if semantic_layout else None
            render_ocr_to_html_multi(all_ocr, out_dir / "layout.html", use_raw_ratios=refine_layout, page_groups=page_groups)
            logger.info("  Layout: layout.html (%d pages)", len(all_ocr))
            satisfaction_dir = out_dir / "satisfaction"
            rendered = render_layout_to_satisfaction(out_dir / "layout.html", satisfaction_dir)
            if rendered:
                logger.info("  Rendered %d page(s) to %s", len(rendered), satisfaction_dir)
        except Exception as e:
            logger.warning("  Layout failed: %s", e)
        if use_xmind:
            try:
                xmind_path = build_xmind(all_ocr, out_dir / f"{safe_name}.xmind", sheet_title=safe_name)
                logger.info("  XMind: %s", xmind_path.name)
            except Exception as e:
                logger.warning("  XMind export failed: %s", e)

        logger.info("  Notebook completed in %.2fs", time.perf_counter() - nb_start)

    logger.info("Done. Output: %s", output_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
