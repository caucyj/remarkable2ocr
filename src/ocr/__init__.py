"""OCR: image -> structured data (Gemini), with cache and confidence."""
from .gemini_ocr import ocr_image, refine_layout_with_gemini, semantic_layout_with_gemini

__all__ = ["ocr_image", "refine_layout_with_gemini", "semantic_layout_with_gemini"]
