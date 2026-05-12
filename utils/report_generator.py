from __future__ import annotations

import io
import tempfile
from datetime import datetime

import numpy as np
from PIL import Image

try:
    from fpdf import FPDF

    FPDF_AVAILABLE = True
except Exception:
    FPDF_AVAILABLE = False


def _to_temp_png(image_obj) -> str:
    if isinstance(image_obj, np.ndarray):
        image = Image.fromarray(image_obj.astype(np.uint8))
    elif isinstance(image_obj, Image.Image):
        image = image_obj
    else:
        raise ValueError("Unsupported image object type.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image.save(tmp.name, format="PNG")
    return tmp.name


def build_pdf_report(report: dict) -> tuple[bytes | None, str]:
    """
    Build a PDF report containing image, predictions, explanations, and XAI panels.
    Returns (pdf_bytes, filename).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.pdf"
    if not FPDF_AVAILABLE:
        return None, filename

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    content_w = 190
    pdf.set_font("Arial", "B", 14)
    pdf.cell(content_w, 10, "Vision Transformer Explainable AI Report", ln=1)

    pdf.set_font("Arial", size=11)
    pdf.multi_cell(content_w, 7, f"Prediction: {report.get('label', '--')}")
    pdf.multi_cell(content_w, 7, f"Confidence: {report.get('confidence', 0.0):.2f}%")
    pdf.multi_cell(content_w, 7, f"Model: {report.get('model_name', '--')} ({report.get('model_mode', '--')})")
    pdf.ln(2)

    explanation = report.get("technical_explanation") or report.get("explanation") or "N/A"
    pdf.set_font("Arial", "B", 12)
    pdf.cell(content_w, 8, "Explanation", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(content_w, 6, explanation)
    pdf.ln(2)

    image_paths: list[tuple[str, str]] = []
    try:
        if report.get("image") is not None:
            image_paths.append(("Original", _to_temp_png(report["image"])))
        if report.get("gradcam_overlay") is not None:
            image_paths.append(("Grad-CAM", _to_temp_png(report["gradcam_overlay"])))
        if report.get("attention_map_overlay") is not None:
            image_paths.append(("Attention Map", _to_temp_png(report["attention_map_overlay"])))
        if report.get("attention_rollout_overlay") is not None:
            image_paths.append(("Attention Rollout", _to_temp_png(report["attention_rollout_overlay"])))
        if report.get("integrated_gradients_overlay") is not None:
            image_paths.append(("Integrated Gradients", _to_temp_png(report["integrated_gradients_overlay"])))

        for title, path in image_paths:
            pdf.set_font("Arial", "B", 11)
            pdf.cell(content_w, 8, title, ln=1)
            pdf.image(path, x=15, w=180)
            pdf.ln(2)
    finally:
        pass

    raw = pdf.output(dest="S")
    if isinstance(raw, str):
        raw = raw.encode("latin1")
    elif isinstance(raw, bytearray):
        raw = bytes(raw)
    elif isinstance(raw, memoryview):
        raw = raw.tobytes()
    elif not isinstance(raw, bytes):
        raw = bytes(raw)
    return raw, filename
