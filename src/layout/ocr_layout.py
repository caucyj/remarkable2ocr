"""
Layout from OCR lines: one div per line, positioned by OCR coordinates; optional alignment guides and links.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

# Padding ratio (0~1); content area [pad, 1-pad]
PADDING_RATIO = 0.05
# Min span for normalization to avoid div-by-zero and spread
MIN_SPAN = 0.15


def _esc(s: str) -> str:
    return html.escape(str(s))


def _normalize_positions(
    ocr_lines: list[dict[str, Any]],
    padding: float,
) -> list[tuple[float, float]]:
    """Normalize positions to [padding, 1-padding] from OCR coordinate range."""
    valid = [
        (float(r.get("x_ratio", 0.5)), float(r.get("y_ratio", 0.5)))
        for r in ocr_lines
        if (r.get("text") or "").strip()
    ]
    if not valid:
        return []
    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, MIN_SPAN)
    span_y = max(max_y - min_y, MIN_SPAN)
    lo = padding
    hi = 1.0 - padding
    out = []
    for x, y in valid:
        nx = lo + (x - min_x) / span_x * (hi - lo)
        ny = lo + (y - min_y) / span_y * (hi - lo)
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        out.append((nx, ny))
    return out


def _percent(v: float) -> str:
    """Convert 0~1 to percentage string."""
    return f"{v * 100:.2f}%"


def render_ocr_to_html(
    ocr_lines: list[dict[str, Any]],
    out_path: Path | str,
    *,
    padding_ratio: float = PADDING_RATIO,
) -> Path:
    """
    Single-page layout: one div per OCR line, absolute position; draggable.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    divs = _build_one_page_divs(ocr_lines, padding_ratio)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>OCR Layout</title>
<style>
  :root {{ font-family: sans-serif; font-size: 14px; color: #222; }}
  .ocr-page {{
    position: relative;
    width: 100%;
    max-width: 720px;
    margin: 0 auto;
    padding: 0;
    aspect-ratio: 3/4;
    max-height: 90vh;
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    box-sizing: border-box;
  }}
  .ocr-line {{
    position: absolute;
    transform: translate(-50%, -50%);
    white-space: nowrap;
    padding: 3px 8px;
    line-height: 1.4;
    cursor: grab;
    user-select: none;
  }}
  .ocr-line:active {{
    cursor: grabbing;
  }}
</style>
</head>
<body>
<div class="ocr-page" id="ocr-page">
{chr(10).join(divs)}
</div>
<script>
(function() {{
  var page = document.getElementById("ocr-page");
  var lines = page.querySelectorAll(".ocr-line");
  var dragging = null;
  var startX, startY, startLeft, startTop;

  function pctToNum(s) {{
    return parseFloat(s) || 0;
  }}
  function numToPct(n) {{
    return (Math.max(0, Math.min(100, n))).toFixed(2) + "%";
  }}

  lines.forEach(function(el) {{
    el.addEventListener("mousedown", function(e) {{
      if (e.button !== 0) return;
      e.preventDefault();
      var rect = page.getBoundingClientRect();
      startX = e.clientX;
      startY = e.clientY;
      startLeft = pctToNum(el.style.left);
      startTop = pctToNum(el.style.top);
      dragging = el;
    }});
  }});

  document.addEventListener("mousemove", function(e) {{
    if (!dragging) return;
    e.preventDefault();
    var rect = page.getBoundingClientRect();
    var dx = (e.clientX - startX) / rect.width * 100;
    var dy = (e.clientY - startY) / rect.height * 100;
    dragging.style.left = numToPct(startLeft + dx);
    dragging.style.top = numToPct(startTop + dy);
  }});

  document.addEventListener("mouseup", function() {{
    dragging = null;
  }});
  document.addEventListener("mouseleave", function() {{
    dragging = null;
  }});
}})();
</script>
</body>
</html>
"""
    out_path.write_text(html_content, encoding="utf-8")
    return out_path


def _build_one_page_divs(
    ocr_lines: list[dict[str, Any]],
    padding_ratio: float,
    *,
    use_raw_ratios: bool = False,
    page_groups: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Build div list for one page; supports shape (box/circle), color, links, optional width/height, and group frames."""
    pad = padding_ratio
    if use_raw_ratios:
        positions = [(float(r.get("x_ratio", 0.5)), float(r.get("y_ratio", 0.5))) for r in ocr_lines if (r.get("text") or "").strip()]
    else:
        positions = _normalize_positions(ocr_lines, pad)
    divs = []
    idx = 0
    index_to_pos: dict[int, tuple[float, float]] = {}
    for i, row in enumerate(ocr_lines):
        text = row.get("text", "").strip()
        if not text:
            continue
        if idx >= len(positions):
            break
        x_norm, y_norm = positions[idx]
        index_to_pos[i] = (x_norm, y_norm)
        idx += 1
    if page_groups and index_to_pos:
        frame_margin = 0.02
        for g in page_groups:
            indices = g.get("indices") or []
            pts = [index_to_pos[k] for k in indices if k in index_to_pos]
            if len(pts) < 2:
                continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            min_x = max(0.0, min(xs) - frame_margin)
            max_x = min(1.0, max(xs) + frame_margin)
            min_y = max(0.0, min(ys) - frame_margin)
            max_y = min(1.0, max(ys) + frame_margin)
            cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
            w, h = max_x - min_x, max_y - min_y
            label = (g.get("label") or "").strip()
            data_attrs = f' data-group-indices="{",".join(str(k) for k in indices)}"'
            if label:
                data_attrs += f' data-group-label="{_esc(label)}"'
            style = f"left:{_percent(cx)};top:{_percent(cy)};width:{_percent(w)};height:{_percent(h)}"
            divs.append(f'<div class="ocr-block ocr-frame"{data_attrs} style="{style}" contenteditable="false">{_esc(label)}</div>')
    idx = 0
    for i, row in enumerate(ocr_lines):
        text = row.get("text", "").strip()
        if not text:
            continue
        if idx >= len(positions):
            break
        x_norm, y_norm = positions[idx]
        idx += 1
        left = _percent(x_norm)
        top = _percent(y_norm)
        shape = row.get("shape") if row.get("shape") in ("box", "circle") else ""
        color = (row.get("color") or "").strip()
        links = row.get("links") or []
        links_str = json.dumps(links, ensure_ascii=False)
        data_attrs = f' data-index="{i}" data-links="{_esc(links_str)}"'
        if shape:
            data_attrs += f' data-shape="{_esc(shape)}"'
        if color:
            data_attrs += f' data-color="{_esc(color)}"'
        color_style = f" color: {_esc(color)};" if color else ""
        size_style = ""
        if row.get("width_ratio") and row.get("height_ratio"):
            w = max(2, min(100, float(row["width_ratio"]) * 100))
            h = max(2, min(100, float(row["height_ratio"]) * 100))
            size_style = f" width: {w:.1f}%; height: {h:.1f}%;"
        content = _esc(text)
        if shape == "box":
            divs.append(
                f'<div class="ocr-block ocr-shape-box"{data_attrs} style="left:{left};top:{top};{color_style}{size_style}" contenteditable="false">{content}</div>'
            )
        elif shape == "circle":
            divs.append(
                f'<div class="ocr-block ocr-shape-circle"{data_attrs} style="left:{left};top:{top};{color_style}{size_style}" contenteditable="false">{content}</div>'
            )
        else:
            divs.append(
                f'<div class="ocr-line"{data_attrs} style="left:{left};top:{top};{color_style}{size_style}" contenteditable="false">{content}</div>'
            )
    return divs


def render_ocr_to_html_multi(
    all_ocr_pages: list[list[dict[str, Any]]],
    out_path: Path | str,
    *,
    padding_ratio: float = PADDING_RATIO,
    use_raw_ratios: bool = False,
    page_groups: list[list[dict[str, Any]]] | None = None,
) -> Path:
    """
    Multi-page layout: one section per page, one .ocr-page per section; draggable.
    When use_raw_ratios is True, x_ratio/y_ratio from OCR are used directly (no normalization) for layout fidelity.
    page_groups: optional list of per-page groups (each group: { "label": str, "indices": list[int] }) for outer frames.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sections = []
    for page_idx, ocr_lines in enumerate(all_ocr_pages):
        if not ocr_lines:
            continue
        groups = (page_groups[page_idx] if page_groups and page_idx < len(page_groups) else None) or []
        divs = _build_one_page_divs(ocr_lines, padding_ratio, use_raw_ratios=use_raw_ratios, page_groups=groups or None)
        if not divs:
            continue
        section_html = f"""<section class="layout-section">
  <h2 class="layout-section-title">Page {page_idx + 1}</h2>
  <div class="ocr-page-wrap">
    <div class="ocr-guides-toggle"><label><input type="checkbox" class="ocr-guides-checkbox"> Guides</label></div>
    <div class="ocr-page" id="ocr-page-{page_idx}" data-page="{page_idx}">
{chr(10).join(divs)}
    </div>
    <svg class="ocr-arrows" id="arrows-{page_idx}" viewBox="0 0 100 100" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg"></svg>
  </div>
</section>"""
        sections.append(section_html)

    if not sections:
        out_path.write_text("<!DOCTYPE html><html><body><p>No OCR data</p></body></html>", encoding="utf-8")
        return out_path

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>OCR Layout</title>
<style>
  :root {{ font-family: sans-serif; font-size: 14px; color: #222; }}
  .layout-section {{ margin-bottom: 2rem; }}
  .layout-section-title {{ font-size: 1rem; margin: 0 0 0.5rem 0; color: #555; }}
  .ocr-page-wrap {{ position: relative; width: 100%; max-width: 720px; margin: 0 auto; aspect-ratio: 3/4; max-height: 90vh; }}
  .ocr-page {{
    position: absolute; left: 0; top: 0; right: 0; bottom: 0;
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    box-sizing: border-box;
  }}
  .ocr-arrows {{ position: absolute; left: 0; top: 0; width: 100%; height: 100%; border-radius: 8px; pointer-events: none; }}
  .ocr-arrows .arrow-paths {{ pointer-events: none; }}
  .ocr-line, .ocr-block {{
    position: absolute;
    transform: translate(-50%, -50%);
    white-space: nowrap;
    padding: 3px 8px;
    line-height: 1.4;
    cursor: grab;
    user-select: none;
    outline: none;
  }}
  .ocr-line:active, .ocr-block:active {{ cursor: grabbing; }}
  .ocr-block.ocr-shape-box {{ border: 2px solid #333; border-radius: 4px; background: rgba(255,255,255,0.9); }}
  .ocr-block.ocr-shape-circle {{ border: 2px solid #333; border-radius: 50%; background: rgba(255,255,255,0.9); padding: 6px 10px; }}
  .save-bar {{ position: fixed; top: 0; left: 0; right: 0; padding: 8px 16px; background: #333; color: #fff; z-index: 1000; }}
  .save-bar button {{ padding: 6px 12px; cursor: pointer; background: #0af; color: #fff; border: none; border-radius: 4px; }}
  .conn-menu {{ position: fixed; z-index: 1001; background: #fff; border: 1px solid #ccc; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); padding: 4px 0; min-width: 140px; }}
  .conn-menu-sub {{ min-width: 180px; }}
  .conn-menu-item {{ padding: 6px 12px; cursor: pointer; font-size: 13px; }}
  .conn-menu-item:hover {{ background: #f0f0f0; }}
  .conn-menu-add {{ border-top: 1px solid #eee; margin-top: 2px; }}
  .ocr-line.selected, .ocr-block.selected {{ outline: 2px solid #0af; outline-offset: 2px; }}
  .ocr-selection-box {{ position: absolute; border: 2px dashed #0af; background: rgba(0,170,255,0.08); pointer-events: none; z-index: 10; }}
  .ocr-guides {{ position: absolute; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none; z-index: 20; }}
  .ocr-guide-line {{ position: absolute; background: none; }}
  .ocr-guide-line.v {{ left: 0; top: 0; bottom: 0; width: 0; border-left: 1px dashed #999; }}
  .ocr-guide-line.h {{ left: 0; top: 0; right: 0; height: 0; border-top: 1px dashed #999; }}
  .ocr-block.ocr-frame {{ border: 2px solid #333; border-radius: 4px; background: rgba(255,255,255,0.5); box-sizing: border-box; min-width: 40px; min-height: 24px; }}
  .ocr-guides-toggle {{ position: absolute; top: 8px; right: 8px; z-index: 30; pointer-events: auto; }}
  .ocr-guides-toggle label {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: #555; cursor: pointer; user-select: none; background: rgba(255,255,255,0.95); padding: 4px 8px; border-radius: 4px; border: 1px solid #ddd; }}
  .ocr-guides-toggle input {{ margin: 0; }}
</style>
</head>
<body>
<div class="save-bar"><button type="button" id="save-layout-btn">Save positions and text to HTML</button></div>
<div style="height: 44px;"></div>
{chr(10).join(sections)}
<script>
(function() {{
  var ns = "http://www.w3.org/2000/svg";
  var DRAG_THRESHOLD = 5;
  var ARROW_STROKE = 0.22;
  var SNAP_THRESHOLD = 1.5;

  function pctToNum(s) {{ return parseFloat(s) || 0; }}
  function numToPct(n) {{ return (Math.max(0, Math.min(100, n))).toFixed(2) + "%"; }}

  function rectToViewBox(rect, wrapRect) {{
    return {{
      left: (rect.left - wrapRect.left) / wrapRect.width * 100,
      right: (rect.right - wrapRect.left) / wrapRect.width * 100,
      top: (rect.top - wrapRect.top) / wrapRect.height * 100,
      bottom: (rect.bottom - wrapRect.top) / wrapRect.height * 100,
      cx: (rect.left + rect.width / 2 - wrapRect.left) / wrapRect.width * 100,
      cy: (rect.top + rect.height / 2 - wrapRect.top) / wrapRect.height * 100
    }};
  }}
  function edgePoints(v) {{
    var cx = (v.left + v.right) / 2, cy = (v.top + v.bottom) / 2;
    return {{ top: [cx, v.top], bottom: [cx, v.bottom], left: [v.left, cy], right: [v.right, cy] }};
  }}
  function pickEndpoints(v1, v2) {{
    var p1 = edgePoints(v1), p2 = edgePoints(v2);
    var cx1 = v1.cx, cy1 = v1.cy, cx2 = v2.cx, cy2 = v2.cy;
    var dx = cx2 - cx1, dy = cy2 - cy1;
    var fromPt, toPt;
    if (Math.abs(dx) >= Math.abs(dy)) {{
      if (dx > 0) {{ fromPt = p1.right; toPt = p2.left; }}
      else {{ fromPt = p1.left; toPt = p2.right; }}
    }} else {{
      if (dy > 0) {{ fromPt = p1.bottom; toPt = p2.top; }}
      else {{ fromPt = p1.top; toPt = p2.bottom; }}
    }}
    return {{ from: fromPt, to: toPt }};
  }}

  function updateArrows(pageEl) {{
    var wrap = pageEl.closest(".ocr-page-wrap");
    var svg = wrap ? wrap.querySelector(".ocr-arrows") : null;
    if (!svg || !wrap) return;
    var pathGroup = svg.querySelector(".arrow-paths") || (function() {{
      var g = document.createElementNS(ns, "g");
    g.setAttribute("class", "arrow-paths");
    g.setAttribute("pointer-events", "none");
    svg.appendChild(g);
    return g;
    }})();
    pathGroup.innerHTML = "";
    var wrapRect = wrap.getBoundingClientRect();
    var byIndex = {{}};
    pageEl.querySelectorAll("[data-index]").forEach(function(el) {{
      var i = parseInt(el.getAttribute("data-index"), 10);
      if (!isNaN(i)) byIndex[i] = el;
    }});
    pageEl.querySelectorAll("[data-links]").forEach(function(fromEl) {{
      var linksStr = fromEl.getAttribute("data-links");
      if (!linksStr) return;
      var links = [];
      try {{ links = JSON.parse(linksStr); }} catch(e) {{ return; }}
      var r1 = fromEl.getBoundingClientRect();
      var v1 = rectToViewBox(r1, wrapRect);
      links.forEach(function(toIndex) {{
        var toEl = byIndex[toIndex];
        if (!toEl) return;
        var r2 = toEl.getBoundingClientRect();
        var v2 = rectToViewBox(r2, wrapRect);
        var pts = pickEndpoints(v1, v2);
        var x1 = pts.from[0], y1 = pts.from[1], x2 = pts.to[0], y2 = pts.to[1];
        x1 = Math.max(0, Math.min(100, x1));
        y1 = Math.max(0, Math.min(100, y1));
        x2 = Math.max(0, Math.min(100, x2));
        y2 = Math.max(0, Math.min(100, y2));
        var cpx1 = x1 + (x2 - x1) * 0.4;
        var cpy1 = y1;
        var cpx2 = x2 - (x2 - x1) * 0.4;
        var cpy2 = y2;
        var d = "M " + x1 + " " + y1 + " C " + cpx1 + " " + cpy1 + ", " + cpx2 + " " + cpy2 + ", " + x2 + " " + y2;
        var path = document.createElementNS(ns, "path");
        path.setAttribute("d", d);
        path.setAttribute("stroke", "#333");
        path.setAttribute("stroke-width", ARROW_STROKE);
        path.setAttribute("fill", "none");
        path.setAttribute("data-from", fromEl.getAttribute("data-index"));
        path.setAttribute("data-to", toIndex);
        pathGroup.appendChild(path);
      }});
    }});
    refreshStaticGuidesForPage(pageEl);
  }}

  var pages = document.querySelectorAll(".ocr-page");
  var dragging = null;
  var pending = null;
  var startX, startY, startLeft, startTop;
  var selectedSet = new Set();
  var boxSelecting = null;

  function setSelection(els) {{
    document.querySelectorAll(".ocr-line.selected, .ocr-block.selected").forEach(function(el) {{ el.classList.remove("selected"); }});
    selectedSet.clear();
    (els || []).forEach(function(el) {{
      if (el && (el.classList.contains("ocr-line") || el.classList.contains("ocr-block"))) {{
        el.classList.add("selected");
        selectedSet.add(el);
      }}
    }});
  }}
  function getSelectedOnPage(page) {{
    var list = [];
    selectedSet.forEach(function(el) {{ if (el.closest(".ocr-page") === page) list.push(el); }});
    return list;
  }}
  function rectsIntersect(r1, r2) {{
    return !(r1.right < r2.left || r1.left > r2.right || r1.bottom < r2.top || r1.top > r2.bottom);
  }}

  function getSnapAndGuides(page, moveEls, proposedDx, proposedDy, pageRect) {{
    var moveSet = new Set();
    moveEls.forEach(function(o) {{ moveSet.add(o.el); }});
    var others = [];
    page.querySelectorAll(".ocr-line, .ocr-block").forEach(function(el) {{
      if (moveSet.has(el)) return;
      var left = pctToNum(el.style.left);
      var top = pctToNum(el.style.top);
      var r = el.getBoundingClientRect();
      var w = r.width / pageRect.width * 100;
      var h = r.height / pageRect.height * 100;
      others.push({{ leftEdge: left - w/2, rightEdge: left + w/2, topEdge: top - h/2, bottomEdge: top + h/2, cx: left, cy: top }});
    }});
    var snapDxCandidates = [];
    var snapDyCandidates = [];
    var guideX = null;
    var guideY = null;
    moveEls.forEach(function(o) {{
      var left = o.startLeft + proposedDx;
      var top = o.startTop + proposedDy;
      var le = left - o.widthPct/2, re = left + o.widthPct/2, te = top - o.heightPct/2, be = top + o.heightPct/2;
      others.forEach(function(ot) {{
        if (Math.abs(le - ot.leftEdge) <= SNAP_THRESHOLD) {{ snapDxCandidates.push({{ dx: ot.leftEdge + o.widthPct/2 - o.startLeft, gx: ot.leftEdge }}); }}
        if (Math.abs(le - ot.rightEdge) <= SNAP_THRESHOLD) {{ snapDxCandidates.push({{ dx: ot.rightEdge + o.widthPct/2 - o.startLeft, gx: ot.rightEdge }}); }}
        if (Math.abs(left - ot.cx) <= SNAP_THRESHOLD) {{ snapDxCandidates.push({{ dx: ot.cx - o.startLeft, gx: ot.cx }}); }}
        if (Math.abs(re - ot.leftEdge) <= SNAP_THRESHOLD) {{ snapDxCandidates.push({{ dx: ot.leftEdge - o.widthPct/2 - o.startLeft, gx: ot.leftEdge }}); }}
        if (Math.abs(re - ot.rightEdge) <= SNAP_THRESHOLD) {{ snapDxCandidates.push({{ dx: ot.rightEdge - o.widthPct/2 - o.startLeft, gx: ot.rightEdge }}); }}
        if (Math.abs(te - ot.topEdge) <= SNAP_THRESHOLD) {{ snapDyCandidates.push({{ dy: ot.topEdge + o.heightPct/2 - o.startTop, gy: ot.topEdge }}); }}
        if (Math.abs(te - ot.bottomEdge) <= SNAP_THRESHOLD) {{ snapDyCandidates.push({{ dy: ot.bottomEdge + o.heightPct/2 - o.startTop, gy: ot.bottomEdge }}); }}
        if (Math.abs(top - ot.cy) <= SNAP_THRESHOLD) {{ snapDyCandidates.push({{ dy: ot.cy - o.startTop, gy: ot.cy }}); }}
        if (Math.abs(be - ot.topEdge) <= SNAP_THRESHOLD) {{ snapDyCandidates.push({{ dy: ot.topEdge - o.heightPct/2 - o.startTop, gy: ot.topEdge }}); }}
        if (Math.abs(be - ot.bottomEdge) <= SNAP_THRESHOLD) {{ snapDyCandidates.push({{ dy: ot.bottomEdge - o.heightPct/2 - o.startTop, gy: ot.bottomEdge }}); }}
      }});
    }});
    var dx = proposedDx, dy = proposedDy;
    if (snapDxCandidates.length > 0) {{
      snapDxCandidates.sort(function(a, b) {{ return Math.abs(a.dx - proposedDx) - Math.abs(b.dx - proposedDx); }});
      dx = snapDxCandidates[0].dx;
      guideX = snapDxCandidates[0].gx;
    }}
    if (snapDyCandidates.length > 0) {{
      snapDyCandidates.sort(function(a, b) {{ return Math.abs(a.dy - proposedDy) - Math.abs(b.dy - proposedDy); }});
      dy = snapDyCandidates[0].dy;
      guideY = snapDyCandidates[0].gy;
    }}
    return {{ dx: dx, dy: dy, guideX: guideX, guideY: guideY }};
  }}

  function showGuides(wrap, guideX, guideY) {{
    var container = wrap.querySelector(".ocr-guides");
    if (!container) {{
      container = document.createElement("div");
      container.className = "ocr-guides";
      wrap.appendChild(container);
    }}
    container.innerHTML = "";
    if (guideX != null) {{
      var v = document.createElement("div");
      v.className = "ocr-guide-line v";
      v.style.left = guideX + "%";
      container.appendChild(v);
    }}
    if (guideY != null) {{
      var h = document.createElement("div");
      h.className = "ocr-guide-line h";
      h.style.top = guideY + "%";
      container.appendChild(h);
    }}
  }}
  function clearAllGuides() {{
    document.querySelectorAll(".ocr-guides").forEach(function(c) {{ c.innerHTML = ""; }});
  }}

  function updateStaticGuides(pageEl) {{
    var wrap = pageEl.closest(".ocr-page-wrap");
    if (!wrap) return;
    var container = wrap.querySelector(".ocr-guides");
    if (!container) {{
      container = document.createElement("div");
      container.className = "ocr-guides";
      wrap.appendChild(container);
    }}
    container.innerHTML = "";
    var step = 12;
    var seenX = {{}}, seenY = {{}};
    for (var x = 0; x <= 100; x += step) {{ seenX[x] = true; }}
    seenX[50] = true;
    for (var y = 0; y <= 100; y += step) {{ seenY[y] = true; }}
    seenY[50] = true;
    Object.keys(seenX).forEach(function(x) {{
      var v = document.createElement("div");
      v.className = "ocr-guide-line v";
      v.style.left = x + "%";
      container.appendChild(v);
    }});
    Object.keys(seenY).forEach(function(y) {{
      var h = document.createElement("div");
      h.className = "ocr-guide-line h";
      h.style.top = y + "%";
      container.appendChild(h);
    }});
  }}

  function refreshStaticGuidesForPage(pageEl) {{
    var wrap = pageEl && pageEl.closest(".ocr-page-wrap");
    if (!wrap) return;
    var cb = wrap.querySelector(".ocr-guides-checkbox");
    if (cb && cb.checked) updateStaticGuides(pageEl);
  }}

  function clearGuidesForWrap(wrap) {{
    var c = wrap && wrap.querySelector(".ocr-guides");
    if (c) c.innerHTML = "";
  }}

  function nextIndex(page) {{
    var max = -1;
    page.querySelectorAll("[data-index]").forEach(function(el) {{
      var i = parseInt(el.getAttribute("data-index"), 10);
      if (!isNaN(i) && i > max) max = i;
    }});
    return max + 1;
  }}

  function bindBlockEvents(page, el) {{
    el.addEventListener("mousedown", function(e) {{
      if (e.button !== 0) return;
      if (e.target !== el && !el.contains(e.target)) return;
      e.preventDefault();
      startX = e.clientX;
      startY = e.clientY;
      startLeft = pctToNum(el.style.left);
      startTop = pctToNum(el.style.top);
      pending = {{ el: el, page: page }};
    }});
    el.addEventListener("dblclick", function(e) {{
      e.preventDefault();
      this.setAttribute("contenteditable", "true");
      this.focus();
      if (window.getSelection) {{
        var r = document.createRange();
        r.selectNodeContents(this);
        window.getSelection().removeAllRanges();
        window.getSelection().addRange(r);
      }}
    }});
    el.addEventListener("blur", function() {{ this.setAttribute("contenteditable", "false"); }});
  }}

  function addDivAt(page, leftPct, topPct, text, shape) {{
    var idx = nextIndex(page);
    var div = document.createElement("div");
    if (shape === "box") div.className = "ocr-block ocr-shape-box";
    else if (shape === "circle") div.className = "ocr-block ocr-shape-circle";
    else div.className = "ocr-line";
    div.setAttribute("data-index", idx);
    div.setAttribute("data-links", "[]");
    div.style.left = numToPct(leftPct);
    div.style.top = numToPct(topPct);
    div.contentEditable = "false";
    div.textContent = text || "New block";
    page.appendChild(div);
    bindBlockEvents(page, div);
    return div;
  }}

  function addFrameAroundSelection(page) {{
    var list = getSelectedOnPage(page);
    if (list.length < 2) return;
    var pr = page.getBoundingClientRect();
    var minL = 100, maxR = 0, minT = 100, maxB = 0;
    list.forEach(function(el) {{
      var l = pctToNum(el.style.left);
      var t = pctToNum(el.style.top);
      var r = el.getBoundingClientRect();
      var w = r.width / pr.width * 100;
      var h = r.height / pr.height * 100;
      minL = Math.min(minL, l - w/2);
      maxR = Math.max(maxR, l + w/2);
      minT = Math.min(minT, t - h/2);
      maxB = Math.max(maxB, t + h/2);
    }});
    var pad = 2;
    minL -= pad; maxR += pad; minT -= pad; maxB += pad;
    var cx = (minL + maxR) / 2;
    var cy = (minT + maxB) / 2;
    var w = maxR - minL;
    var h = maxB - minT;
    var idx = nextIndex(page);
    var frame = document.createElement("div");
    frame.className = "ocr-block ocr-shape-box ocr-frame";
    frame.setAttribute("data-index", idx);
    frame.setAttribute("data-links", "[]");
    frame.style.left = numToPct(cx);
    frame.style.top = numToPct(cy);
    frame.style.width = w + "%";
    frame.style.height = h + "%";
    frame.contentEditable = "false";
    frame.textContent = "";
    page.insertBefore(frame, page.firstChild);
    bindBlockEvents(page, frame);
    setSelection([]);
    updateArrows(page);
    return frame;
  }}

  function pointInSelectionBox(clientX, clientY, page) {{
    var list = getSelectedOnPage(page);
    if (list.length < 2) return false;
    var pr = page.getBoundingClientRect();
    var minL = 1e9, maxR = -1e9, minT = 1e9, maxB = -1e9;
    list.forEach(function(el) {{
      var r = el.getBoundingClientRect();
      minL = Math.min(minL, r.left);
      maxR = Math.max(maxR, r.right);
      minT = Math.min(minT, r.top);
      maxB = Math.max(maxB, r.bottom);
    }});
    var x = clientX, y = clientY;
    return x >= minL && x <= maxR && y >= minT && y <= maxB;
  }}

  function showConnMenu(div, page, clientX, clientY) {{
    var menu = document.createElement("div");
    menu.className = "conn-menu";
    menu.style.left = clientX + "px";
    menu.style.top = clientY + "px";

    var toRemove = selectedSet.size > 0 ? Array.from(selectedSet).filter(function(el) {{ return el.closest(".ocr-page") === page; }}) : [div];
    var delItem = document.createElement("div");
    delItem.className = "conn-menu-item";
    delItem.textContent = toRemove.length > 1 ? "Delete selected (" + toRemove.length + ")" : "Delete this block";
    delItem.addEventListener("click", function() {{
      var removedIndices = {{}};
      toRemove.forEach(function(el) {{
        var i = parseInt(el.getAttribute("data-index"), 10);
        if (!isNaN(i)) removedIndices[i] = true;
        selectedSet.delete(el);
        if (el.parentNode) el.parentNode.removeChild(el);
      }});
      page.querySelectorAll("[data-links]").forEach(function(el) {{
        var links = [];
        try {{ links = JSON.parse(el.getAttribute("data-links") || "[]"); }} catch(x) {{}}
        links = links.filter(function(i) {{ return !removedIndices[i]; }});
        el.setAttribute("data-links", JSON.stringify(links));
      }});
      setSelection([]);
      updateArrows(page);
      if (menu.parentNode) menu.parentNode.removeChild(menu);
      document.removeEventListener("click", closeMenu);
    }});
    menu.appendChild(delItem);

    var copyItem = document.createElement("div");
    copyItem.className = "conn-menu-item";
    copyItem.textContent = toRemove.length > 1 ? "Copy text (" + toRemove.length + " blocks)" : "Copy text";
    copyItem.addEventListener("click", function() {{
      var parts = [];
      toRemove.forEach(function(el) {{ parts.push((el.innerText || el.textContent || "").trim()); }});
      var text = parts.join("\\n");
      if (navigator.clipboard && navigator.clipboard.writeText) {{
        navigator.clipboard.writeText(text).then(function() {{ if (menu.parentNode) menu.parentNode.removeChild(menu); document.removeEventListener("click", closeMenu); }}).catch(function() {{}});
      }} else {{
        var ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        try {{ document.execCommand("copy"); }} catch(x) {{}}
        document.body.removeChild(ta);
        if (menu.parentNode) menu.parentNode.removeChild(menu);
        document.removeEventListener("click", closeMenu);
      }}
    }});
    menu.appendChild(copyItem);

    if (div.classList.contains("ocr-shape-box") || div.classList.contains("ocr-shape-circle")) {{
      var noBorderItem = document.createElement("div");
      noBorderItem.className = "conn-menu-item";
      noBorderItem.textContent = "Remove border";
      noBorderItem.addEventListener("click", function() {{
        div.classList.remove("ocr-shape-box", "ocr-shape-circle");
        div.classList.add("ocr-line");
        div.removeAttribute("data-shape");
        if (menu.parentNode) menu.parentNode.removeChild(menu);
        document.removeEventListener("click", closeMenu);
      }});
      menu.appendChild(noBorderItem);
    }}

    if (selectedSet.size >= 2 && selectedSet.has(div)) {{
      var frameItem = document.createElement("div");
      frameItem.className = "conn-menu-item";
      frameItem.textContent = "Add group frame";
      frameItem.addEventListener("click", function() {{
        addFrameAroundSelection(page);
        if (menu.parentNode) menu.parentNode.removeChild(menu);
        document.removeEventListener("click", closeMenu);
      }});
      menu.appendChild(frameItem);
    }}

    var linksStr = div.getAttribute("data-links");
    var links = [];
    if (linksStr) {{ try {{ links = JSON.parse(linksStr); }} catch(x) {{}} }}
    var myIndex = parseInt(div.getAttribute("data-index"), 10);
    var byIndex = {{}};
    page.querySelectorAll("[data-index]").forEach(function(el2) {{
      var i = parseInt(el2.getAttribute("data-index"), 10);
      if (!isNaN(i)) byIndex[i] = el2;
    }});
    if (links.length > 0) {{
      links.forEach(function(toIdx) {{
        var toEl = byIndex[toIdx];
        var label = (toEl && toEl.textContent) ? toEl.textContent.slice(0, 20) : ("#" + toIdx);
        var item = document.createElement("div");
        item.className = "conn-menu-item";
        item.textContent = "Remove link to " + label;
        item.addEventListener("click", function() {{
          var cur = [];
          try {{ cur = JSON.parse(div.getAttribute("data-links") || "[]"); }} catch(x) {{}}
          cur = cur.filter(function(i) {{ return i !== toIdx; }});
          div.setAttribute("data-links", JSON.stringify(cur));
          updateArrows(page);
          if (menu.parentNode) menu.parentNode.removeChild(menu);
          document.removeEventListener("click", closeMenu);
        }});
        menu.appendChild(item);
      }});
      var allItem = document.createElement("div");
      allItem.className = "conn-menu-item";
      allItem.textContent = "Remove all links";
      allItem.addEventListener("click", function() {{
        div.setAttribute("data-links", "[]");
        updateArrows(page);
        if (menu.parentNode) menu.parentNode.removeChild(menu);
        document.removeEventListener("click", closeMenu);
      }});
      menu.appendChild(allItem);
    }}
    var addItem = document.createElement("div");
    addItem.className = "conn-menu-item conn-menu-add";
    addItem.textContent = "Add link →";
    addItem.addEventListener("click", function(e) {{
      e.stopPropagation();
      if (menu.parentNode) menu.parentNode.removeChild(menu);
      document.removeEventListener("click", closeMenu);
      var pageEl = div.closest(".ocr-page");
      if (!pageEl) return;
      var sub = document.createElement("div");
      sub.className = "conn-menu conn-menu-sub";
      sub.style.left = (clientX + 160) + "px";
      sub.style.top = clientY + "px";
      var curLinks = [];
      try {{ curLinks = JSON.parse(div.getAttribute("data-links") || "[]"); }} catch(x) {{}}
      pageEl.querySelectorAll("[data-index]").forEach(function(el2) {{
        var toIdx = parseInt(el2.getAttribute("data-index"), 10);
        if (isNaN(toIdx) || toIdx === myIndex) return;
        var label = (el2.textContent || "").trim().slice(0, 24) || ("#" + toIdx);
        if (curLinks.indexOf(toIdx) >= 0) label = "✓ " + label;
        var item = document.createElement("div");
        item.className = "conn-menu-item";
        item.textContent = label;
        item.addEventListener("click", function() {{
          var cur = [];
          try {{ cur = JSON.parse(div.getAttribute("data-links") || "[]"); }} catch(x) {{}}
          if (cur.indexOf(toIdx) < 0) cur.push(toIdx);
          div.setAttribute("data-links", JSON.stringify(cur));
          updateArrows(div.closest(".ocr-page"));
          if (sub.parentNode) sub.parentNode.removeChild(sub);
          document.removeEventListener("click", closeSub);
        }});
        sub.appendChild(item);
      }});
      if (sub.childNodes.length === 0) {{
        var empty = document.createElement("div");
        empty.className = "conn-menu-item";
        empty.textContent = "(No other blocks on this page)";
        sub.appendChild(empty);
      }}
      document.body.appendChild(sub);
      function closeSub(ev) {{
        if (!sub.parentNode) return;
        if (!sub.contains(ev.target)) {{
          sub.parentNode.removeChild(sub);
          document.removeEventListener("click", closeSub);
        }}
      }}
      setTimeout(function() {{ document.addEventListener("click", closeSub); }}, 0);
    }});
    menu.appendChild(addItem);
    document.body.appendChild(menu);
    function closeMenu(ev) {{
      if (!menu.parentNode) return;
      if (!menu.contains(ev.target)) {{
        menu.parentNode.removeChild(menu);
        document.removeEventListener("click", closeMenu);
      }}
    }}
    setTimeout(function() {{ document.addEventListener("click", closeMenu); }}, 0);
  }}

  function showEmptyMenu(wrap, page, clientX, clientY) {{
    var menu = document.createElement("div");
    menu.className = "conn-menu";
    menu.style.left = clientX + "px";
    menu.style.top = clientY + "px";
    var addBlockItem = document.createElement("div");
    addBlockItem.className = "conn-menu-item";
    addBlockItem.textContent = "Add text block here";
    addBlockItem.addEventListener("click", function() {{
      var wr = wrap.getBoundingClientRect();
      var leftPct = (clientX - wr.left) / wr.width * 100;
      var topPct = (clientY - wr.top) / wr.height * 100;
      var newEl = addDivAt(page, leftPct, topPct, "New block", "");
      updateArrows(page);
      if (menu.parentNode) menu.parentNode.removeChild(menu);
      document.removeEventListener("click", closeMenu);
      newEl.setAttribute("contenteditable", "true");
      newEl.focus();
      if (window.getSelection) {{
        var r = document.createRange();
        r.selectNodeContents(newEl);
        window.getSelection().removeAllRanges();
        window.getSelection().addRange(r);
      }}
    }});
    menu.appendChild(addBlockItem);
    if (selectedSet.size >= 2 && pointInSelectionBox(clientX, clientY, page)) {{
      var frameItem = document.createElement("div");
      frameItem.className = "conn-menu-item";
      frameItem.textContent = "Add group frame";
      frameItem.addEventListener("click", function() {{
        addFrameAroundSelection(page);
        if (menu.parentNode) menu.parentNode.removeChild(menu);
        document.removeEventListener("click", closeMenu);
      }});
      menu.appendChild(frameItem);
    }}
    document.body.appendChild(menu);
    function closeMenu(ev) {{
      if (!menu.parentNode) return;
      if (!menu.contains(ev.target)) {{
        menu.parentNode.removeChild(menu);
        document.removeEventListener("click", closeMenu);
      }}
    }}
    setTimeout(function() {{ document.addEventListener("click", closeMenu); }}, 0);
  }}

  document.addEventListener("contextmenu", function(e) {{
    var wrap = null;
    var el = e.target;
    while (el && el !== document.body) {{
      if (el.classList && el.classList.contains("ocr-page-wrap")) {{ wrap = el; break; }}
      el = el.parentNode;
    }}
    if (!wrap) return;
    var div = null;
    var els = document.elementsFromPoint ? document.elementsFromPoint(e.clientX, e.clientY) : [];
    if (els.length === 0 && document.elementFromPoint) {{
      var one = document.elementFromPoint(e.clientX, e.clientY);
      if (one) els = [one];
    }}
    if (els.length === 0) els = [e.target];
    for (var i = 0; i < els.length; i++) {{
      var x = els[i];
      if (x.closest && x.closest(".ocr-arrows")) continue;
      if (x.classList && (x.classList.contains("ocr-line") || x.classList.contains("ocr-block"))) {{ div = x; break; }}
      if (x.closest) {{ var d = x.closest(".ocr-line, .ocr-block"); if (d) {{ div = d; break; }} }}
    }}
    if (div && div.closest(".ocr-page-wrap") === wrap) {{
      e.preventDefault();
      e.stopPropagation();
      var page = div.closest(".ocr-page");
      showConnMenu(div, page, e.clientX, e.clientY);
    }} else if (wrap) {{
      var page = wrap.querySelector(".ocr-page");
      if (page) {{
        e.preventDefault();
        e.stopPropagation();
        showEmptyMenu(wrap, page, e.clientX, e.clientY);
      }}
    }}
  }}, true);

  pages.forEach(function(page) {{
    page.addEventListener("mousedown", function(e) {{
      if (e.button !== 0) return;
      if (e.target !== page) return;
      e.preventDefault();
      var wrap = page.closest(".ocr-page-wrap");
      var wr = wrap.getBoundingClientRect();
      var sx = e.clientX - wr.left;
      var sy = e.clientY - wr.top;
      var box = document.createElement("div");
      box.className = "ocr-selection-box";
      box.style.left = sx + "px";
      box.style.top = sy + "px";
      box.style.width = "0";
      box.style.height = "0";
      wrap.appendChild(box);
      boxSelecting = {{ wrap: wrap, page: page, startX: e.clientX, startY: e.clientY, box: box }};
    }});
    var nodes = page.querySelectorAll(".ocr-line, .ocr-block");
    nodes.forEach(function(el) {{
      el.addEventListener("mousedown", function(e) {{
        if (e.button !== 0) return;
        if (e.target !== el && !el.contains(e.target)) return;
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        startLeft = pctToNum(el.style.left);
        startTop = pctToNum(el.style.top);
        pending = {{ el: el, page: page }};
      }});
      el.addEventListener("dblclick", function(e) {{
        e.preventDefault();
        this.setAttribute("contenteditable", "true");
        this.focus();
        if (window.getSelection) {{
          var r = document.createRange();
          r.selectNodeContents(this);
          window.getSelection().removeAllRanges();
          window.getSelection().addRange(r);
        }}
      }});
      el.addEventListener("blur", function() {{
        this.setAttribute("contenteditable", "false");
      }});
    }});
    updateArrows(page);
  }});

  document.querySelectorAll(".ocr-guides-checkbox").forEach(function(cb) {{
    cb.addEventListener("change", function() {{
      var wrap = this.closest(".ocr-page-wrap");
      var page = wrap && wrap.querySelector(".ocr-page");
      if (this.checked) updateStaticGuides(page);
      else clearGuidesForWrap(wrap);
    }});
  }});

  document.addEventListener("mousemove", function(e) {{
    if (boxSelecting) {{
      var wr = boxSelecting.wrap.getBoundingClientRect();
      var minX = Math.min(boxSelecting.startX, e.clientX);
      var maxX = Math.max(boxSelecting.startX, e.clientX);
      var minY = Math.min(boxSelecting.startY, e.clientY);
      var maxY = Math.max(boxSelecting.startY, e.clientY);
      boxSelecting.box.style.left = (minX - wr.left) + "px";
      boxSelecting.box.style.top = (minY - wr.top) + "px";
      boxSelecting.box.style.width = (maxX - minX) + "px";
      boxSelecting.box.style.height = (maxY - minY) + "px";
      return;
    }}
    if (pending && !dragging) {{
      var dx = e.clientX - startX;
      var dy = e.clientY - startY;
      if (dx*dx + dy*dy > DRAG_THRESHOLD * DRAG_THRESHOLD) {{
        dragging = pending;
        pending = null;
        var page = dragging.page;
        var moveEls = [];
        if (selectedSet.has(dragging.el)) {{
          getSelectedOnPage(page).forEach(function(el) {{
            var r = el.getBoundingClientRect();
            var pr = page.getBoundingClientRect();
            moveEls.push({{ el: el, startLeft: pctToNum(el.style.left), startTop: pctToNum(el.style.top), widthPct: r.width / pr.width * 100, heightPct: r.height / pr.height * 100 }});
          }});
        }} else {{
          setSelection([dragging.el]);
          var el = dragging.el;
          var r = el.getBoundingClientRect();
          var pr = page.getBoundingClientRect();
          moveEls = [{{ el: el, startLeft: pctToNum(el.style.left), startTop: pctToNum(el.style.top), widthPct: r.width / pr.width * 100, heightPct: r.height / pr.height * 100 }}];
        }}
        dragging.moveEls = moveEls;
        startX = e.clientX;
        startY = e.clientY;
      }}
    }}
    if (!dragging) return;
    e.preventDefault();
    var page = dragging.page;
    var rect = page.getBoundingClientRect();
    var proposedDx = (e.clientX - startX) / rect.width * 100;
    var proposedDy = (e.clientY - startY) / rect.height * 100;
    var snap = getSnapAndGuides(page, dragging.moveEls, proposedDx, proposedDy, rect);
    var dx = snap.dx, dy = snap.dy;
    dragging.moveEls.forEach(function(o) {{
      o.el.style.left = numToPct(o.startLeft + dx);
      o.el.style.top = numToPct(o.startTop + dy);
    }});
    var wrap = page.closest(".ocr-page-wrap");
    showGuides(wrap, snap.guideX, snap.guideY);
    updateArrows(page);
  }});

  document.addEventListener("mouseup", function(e) {{
    if (boxSelecting) {{
      var boxRect = boxSelecting.box.getBoundingClientRect();
      var hits = [];
      boxSelecting.page.querySelectorAll(".ocr-line, .ocr-block").forEach(function(el) {{
        if (rectsIntersect(el.getBoundingClientRect(), boxRect)) hits.push(el);
      }});
      setSelection(hits);
      if (boxSelecting.box.parentNode) boxSelecting.box.parentNode.removeChild(boxSelecting.box);
      boxSelecting = null;
    }}
    if (pending && !dragging) {{
      if (e.button === 0) {{
        if (e.ctrlKey || e.metaKey) {{
          if (selectedSet.has(pending.el)) {{
            selectedSet.delete(pending.el);
            pending.el.classList.remove("selected");
          }} else {{
            selectedSet.add(pending.el);
            pending.el.classList.add("selected");
          }}
        }} else setSelection([pending.el]);
      }}
    }}
    var draggedPage = dragging ? dragging.page : null;
    dragging = null;
    pending = null;
    clearAllGuides();
    if (draggedPage) refreshStaticGuidesForPage(draggedPage);
    document.querySelectorAll(".ocr-guides-checkbox:checked").forEach(function(cb) {{
      var wrap = cb.closest(".ocr-page-wrap");
      var page = wrap && wrap.querySelector(".ocr-page");
      if (page) updateStaticGuides(page);
    }});
  }});
  document.addEventListener("mouseleave", function() {{ dragging = null; pending = null; clearAllGuides(); document.querySelectorAll(".ocr-guides-checkbox:checked").forEach(function(cb) {{ var wrap = cb.closest(".ocr-page-wrap"); var page = wrap && wrap.querySelector(".ocr-page"); if (page) updateStaticGuides(page); }}); if (boxSelecting) {{ if (boxSelecting.box.parentNode) boxSelecting.box.parentNode.removeChild(boxSelecting.box); boxSelecting = null; }} }});

  document.getElementById("save-layout-btn").addEventListener("click", function() {{
    var html = "<!DOCTYPE html>\\n" + document.documentElement.outerHTML;
    var blob = new Blob([html], {{ type: "text/html;charset=utf-8" }});
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "layout.html";
    a.click();
    URL.revokeObjectURL(a.href);
  }});
}})();
</script>
</body>
</html>
"""
    out_path.write_text(html_content, encoding="utf-8")
    return out_path

