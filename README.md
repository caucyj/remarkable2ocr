# remarkable2ocr

Turn reMarkable handwritten notes (or any note image) into an **editable layout**: scan notebooks or camera images → OCR with Gemini → interactive HTML with draggable blocks, connectors, and alignment guides.

**Install to run:** clone → `python -m venv .venv` → `pip install -r requirements.txt` → copy `.env.example` to `.env` and set `GOOGLE_API_KEY` → `python main.py`.

---

## Background

- **Input:** Notes from a [reMarkable](https://remarkable.com/) tablet (xochitl data) or photos of handwritten pages (e.g. in `data/xochitl/camera/<project>/`).
- **Goal:** Preserve structure (lines, boxes, arrows, colors) and make it editable in the browser: move blocks, edit text, add/remove links, group with frames, copy text.
- **Flow:** Raw data → page images → OCR (text + position + links/shape/color) → optional semantic layout (stricter links + groups) → cached JSON → multi-page `layout.html` with drag-and-drop, connectors, and group frames.

---

## Architecture

The pipeline is split into three modules under `src/`, plus config:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  main.py                                                                │
│  (orchestrates: optional --pull → list/camera → render → OCR → layout)  │
└─────────────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────────────┐
│  src/remarkable │  │   src/ocr       │  │  src/layout                  │
│  ─────────────  │  │  ─────────────  │  │  ──────────────────────────  │
│  • list_notebook│  │  • ocr_image()  │  │  • render_ocr_to_html_multi  │
│  • render_*_png │  │    (Gemini API, │  │    → layout.html (divs,      │
│  • pull_xochitl │  │     cache JSON) │  │      links, group frames,    │
│                 │  │  • semantic_*   │  │      guides)                 │
│  data/xochitl   │  │    (links only  │  │  • write_ocr_preview_html    │
│  → pages/*.png  │  │     when arrow  │  │  • render_ocr_overlay        │
│                 │  │     in image;   │  │  (+ chart schema / SVG       │
│                 │  │     groups for  │  │   for semantic parsing)      │
│                 │  │     oval/box)   │  │                              │
│                 │  │  page PNG →     │  │                              │
│                 │  │  ocr/*.json     │  │                              │
└─────────────────┘  └─────────────────┘  └──────────────────────────────┘
         │                    │                    │
         └────────────────────┴────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  src/config       │
                    │  .env, DATA_DIR,  │
                    │  GOOGLE_API_KEY,  │
                    │  REMARKABLE_*     │
                    └───────────────────┘
```


| Module         | Role                      | Input → Output                                                                                                                                                                                                                  |
| -------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **remarkable** | Device data and rendering | `data/xochitl` (or device via `--pull`) → `output/<name>/pages/*.png`                                                                                                                                                           |
| **ocr**        | Handwriting recognition   | Page image → Gemini → `output/<name>/ocr/page_*.json` (text, y/x ratio, shape, color, confidence). With `--semantic-layout`: adds links only where the image has explicit arrows/lines, and groups (oval/box) for outer frames. |
| **layout**     | Editable UI               | OCR JSON (+ optional per-page groups) → `layout.html` (draggable blocks, connectors, group frames), `.debug/ocr_preview.html`, `ocr_overlay_*.png`                                                                              |
| **config**     | Environment               | `.env` → `DATA_DIR`, `GOOGLE_API_KEY`, `REMARKABLE_HOST`, etc.                                                                                                                                                                  |


- **Notebook mode:** `list_notebooks()` → for each notebook, `render_notebook_pages()` → for each page, `ocr_image()` → `render_ocr_to_html_multi(all_ocr)`.
- **Camera mode:** One or more images in `data/xochitl/camera/<project>/` → same OCR + layout pipeline → `output/<project>/`.

---

## Quick start

```bash
git clone https://github.com/caucyj/remarkable2ocr.git && cd remarkable2ocr
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=your_gemini_api_key
python main.py
```

## Run

From the project root:

```bash
python main.py
```

- **Default:** Uses local OCR cache under `output/<notebook>/ocr/page_*.json` when present; no Gemini API call for cached pages.
- **Force re-OCR:** `python main.py --no-cache`
- **Pull from reMarkable then process:** `python main.py --pull` (fails if device not connected)
- **Pull + no cache:** `python main.py --pull --no-cache`
- **Camera image(s):** Put image(s) in `data/xochitl/camera/<name>/` (e.g. `.jpg` or `.png`), then:
  ```bash
  python main.py --camera <name>
  ```
  Output goes to `output/<name>/`. Optional: `--camera <name> --no-cache`
- **Export to XMind mind map:** Add `--xmind` to also generate `project_name.xmind` (or `notebook_name.xmind`) from the OCR layout; link relationships become parent-child in the mind map.
  ```bash
  python main.py --camera test_camera --xmind
  python main.py --xmind
  ```
- **Process a single project (notebook):** `python main.py --project PROJECT_NAME` processes only the notebook whose safe name equals `PROJECT_NAME`. If no such notebook exists in `data/xochitl` and there is no OCR cache under `output/PROJECT_NAME/`, the program exits with code 1 and a friendly message suggesting to check names or use `--camera PROJECT_NAME` for a camera project.
- **Semantic layout (links + groups):** With `--semantic-layout`, Gemini analyzes the note image to:
  - **Links:** Add connector arrows only between blocks that are **explicitly connected in the drawing** (e.g. an arrow from A to B). Blocks in the same section but without a drawn line (e.g. “Implementation” and “Code risk”) are not linked.
  - **Groups:** Detect blocks inside one oval or box and output a **group**; the layout then draws an **outer frame** around those blocks (e.g. HTTPS, Socket, SSE with label “Protocol”).  
  Semantic result is cached in `ocr/page_*_semantic.json` (format: `{ "items": [...], "groups": [{ "label": "...", "indices": [...] }] }`). Re-run with `--no-cache` to refresh after prompt changes.
- **Refine layout to match a reference image:** `--refine-layout` asks Gemini to adjust block positions (x/y ratio, width/height) to better match the note. Use `--refine-to <path>` to point to a reference image (default: `output/.../satisfaction/expected.png`). Refined JSON is cached in `ocr/page_*_refined*.json`.

## Environment

- Python 3.10+
- Create `.env` in project root (see `.env.example`):
  - `GOOGLE_API_KEY` — required for Gemini OCR (optional if using cache only)
  - `DATA_DIR` — xochitl data directory (default `data/xochitl`)
  - `REMARKABLE_HOST` — device host for `--pull` (default `10.11.99.1`)
  - `REMARKABLE_USER` — SSH user (default `root`)
  - `REMARKABLE_XOCHITL_PATH` — path on device (default `/home/root/.local/share/remarkable/xochitl`)
- Install: `pip install -r requirements.txt`

For `--pull`, ensure the reMarkable is on the same network (e.g. USB or Wi‑Fi) and SSH works (`ssh root@10.11.99.1`). Install `rsync` if missing.

## Output

- `output/<notebook_or_project>/pages/` — page PNGs
- `output/<notebook_or_project>/ocr/` — OCR JSON cache (`page_0.json`, …). With `--semantic-layout`: `page_*_semantic.json` (items + groups). With `--refine-layout`: `page_*_refined*.json`.
- `output/<notebook_or_project>/layout.html` — multi-page layout (draggable divs, connectors from semantic links, group frames for oval/box groups, alignment guides)
- `output/<notebook_or_project>/.debug/` — `ocr_preview.html`, `ocr_overlay_*.png`

## Testing

Automated tests run on every push and pull request via [GitHub Actions](.github/workflows/test.yml). To run locally:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

**Layout similarity report** (optional): compare `satisfaction/render_0.png` with `render_after_modify.png` and `expected.png` (pixel/edge SSIM, optional position similarity and Gemini evaluation):

```bash
python -m tests.report_similarity [project_dir]
# Default project_dir: output/Testing
```

## See also

- [awesome-reMarkable](https://github.com/reHackable/awesome-reMarkable) — A curated list of projects related to the reMarkable tablet (APIs, cloud tools, GUI clients, templates, and more). Useful for discovering other tools and integrations.

