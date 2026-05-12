from __future__ import annotations

import base64
import hashlib
import io
import os
import sys
import threading
import time
from dataclasses import dataclass

try:
    import av
    import cv2
    from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer
    STREAMLIT_WEBRTC_AVAILABLE = True
except ImportError:
    av = None
    cv2 = None
    WebRtcMode = None
    VideoProcessorBase = object
    webrtc_streamer = None
    STREAMLIT_WEBRTC_AVAILABLE = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.prediction_service import (
    explain_image,
    generate_gradcam_for_class,
    predict_image,
    preload_pretrained_model,
)
from utils.report_generator import build_pdf_report
from utils.stress_test import StressConfig, apply_stress_transform, summarize_focus_shift
from utils.uncertainty import analyze_prediction_stability

try:
    from database import delete_prediction_history, fetch_recent_history, save_feedback, save_prediction_history

    MONGO_READY = True
except Exception:
    MONGO_READY = False


ICON_PATH = os.path.join(PROJECT_ROOT, "static", "images", "icon.png")
st.set_page_config(
    page_title="Vision Transformer Explainable AI",
    layout="wide",
    page_icon=Image.open(ICON_PATH) if os.path.exists(ICON_PATH) else None,
)


@st.cache_resource
def _preload_model():
    preload_pretrained_model()
    return True


_preload_model()


@dataclass
class LiveState:
    lock: threading.Lock
    label: str = "--"
    confidence: float = 0.0
    status: str = "OFF"
    last_frame_rgb: any = None
    model_mode: str = "pretrained"
    interval_sec: float = 2.5
    frame_skip: int = 10
    device: str = "--"
    fp16_enabled: bool = False
    top_predictions: list[dict] | None = None


@st.cache_resource
def get_live_state() -> LiveState:
    return LiveState(lock=threading.Lock())


LIVE_STATE = get_live_state()


if STREAMLIT_WEBRTC_AVAILABLE:
    class WebcamPredictor(VideoProcessorBase):
        def __init__(self):
            self.last_pred_at = 0.0
            self.frame_count = 0

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            bgr = frame.to_ndarray(format="bgr24")
            if bgr is None or bgr.size == 0:
                with LIVE_STATE.lock:
                    LIVE_STATE.status = "No valid frame"
                return frame

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.frame_count += 1

            with LIVE_STATE.lock:
                LIVE_STATE.last_frame_rgb = rgb.copy()
                model_mode = LIVE_STATE.model_mode
                interval = LIVE_STATE.interval_sec
                frame_skip = max(1, int(LIVE_STATE.frame_skip))
                status = LIVE_STATE.status

            now = time.time()
            should_run = (
                not str(status).startswith("OFF")
                and self.frame_count % frame_skip == 0
                and (now - self.last_pred_at) >= interval
            )
            if should_run:
                try:
                    pred = predict_image(
                        image=Image.fromarray(rgb),
                        model_mode=model_mode,
                        resize_224=True,
                        return_attentions=False,
                    )
                    with LIVE_STATE.lock:
                        LIVE_STATE.label = pred["label"]
                        LIVE_STATE.confidence = pred["confidence"]
                        LIVE_STATE.status = f"ON ({pred['model_mode']})"
                        LIVE_STATE.device = pred.get("device", "--")
                        LIVE_STATE.fp16_enabled = bool(pred.get("fp16_enabled", False))
                        LIVE_STATE.top_predictions = pred.get("top_predictions", [])
                    self.last_pred_at = now
                except Exception:
                    with LIVE_STATE.lock:
                        LIVE_STATE.status = "Prediction error"

            with LIVE_STATE.lock:
                text_label = LIVE_STATE.label
                text_conf = LIVE_STATE.confidence

            overlay = bgr.copy()
            cv2.rectangle(overlay, (8, 8), (520, 82), (15, 23, 42), -1)
            cv2.putText(overlay, f"Prediction: {text_label}", (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(overlay, f"Confidence: {text_conf * 100:.2f}%", (16, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (165, 243, 252), 2)
            return av.VideoFrame.from_ndarray(overlay, format="bgr24")
else:
    WebcamPredictor = None


def set_styles():
    dark_mode = bool(st.session_state.get("streamlit_dark_mode", False))
    app_bg = "#0f172a" if dark_mode else "#f5f7fb"
    text_color = "#f9fafb" if dark_mode else "#111827"
    card_bg = "#111827" if dark_mode else "#ffffff"
    card_border = "#334155" if dark_mode else "#dbe2f0"
    input_bg = "#1e293b" if dark_mode else "#ffffff"
    input_text = "#f9fafb" if dark_mode else "#111827"
    button_bg = "#2563eb" if dark_mode else "#3b82f6"
    uploader_bg = "#1e293b" if dark_mode else "#eff6ff"
    toggle_bg = "#1e293b" if dark_mode else "#f8fafc"
    alert_bg = "#1e293b" if dark_mode else "#eff6ff"
    status_bg = "#1e293b" if dark_mode else "#dbeafe"
    status_text = "#e2e8f0" if dark_mode else "#1e3a8a"
    header_bg = "#e2e8f0" if dark_mode is False else "#111827"
    header_text = "#0f172a" if dark_mode is False else "#f9fafb"
    css = """
<style>
.stApp {
    background: __APP_BG__;
    color: __TEXT_COLOR__ !important;
}
.stApp,
.stApp p,
.stApp span,
.stApp label,
.stApp li,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: __TEXT_COLOR__ !important;
}
[data-testid="stSidebar"] {
    background: __CARD_BG__ !important;
    border-right: 1px solid __CARD_BORDER__ !important;
}
[data-testid="stSidebar"] * {
    color: __TEXT_COLOR__ !important;
}
header[data-testid="stHeader"] {
    background: __HEADER_BG__ !important;
    border-bottom: 1px solid __CARD_BORDER__ !important;
}
header[data-testid="stHeader"] * {
    color: __HEADER_TEXT__ !important;
}
[data-testid="stToolbar"] * {
    color: __HEADER_TEXT__ !important;
}
[data-testid="stDeployButton"] button,
[data-testid="stBaseButton-headerNoPadding"] {
    background: __BUTTON_BG__ !important;
    color: #ffffff !important;
    border: none !important;
}
.block-container {
    width: 95%;
    max-width: 1500px;
    margin: 0 auto;
    padding: 20px 40px;
}
.hero {
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 18px;
    background: linear-gradient(135deg, #1e3a8a, #1d4ed8 70%, #3b82f6);
    color: white !important;
}
.hero * { color: white !important; }
.dashboard-shell {
    display: flex;
    gap: 20px;
    align-items: flex-start;
}
.control-panel {
    flex: 0 0 25%;
    position: sticky !important;
    top: 84px;
    align-self: flex-start;
    max-height: calc(100vh - 110px);
    overflow-y: auto;
    padding-right: 4px;
}
.dashboard-shell div[data-testid="column"]:first-child {
    position: sticky !important;
    top: 84px;
    align-self: flex-start;
}
.sidebar-controls {
    padding-bottom: 20px;
}
.sidebar-controls * {
    color: __TEXT_COLOR__ !important;
}
.content-panel {
    flex: 1 1 75%;
    min-width: 0;
}
.card {
    background: __CARD_BG__;
    border-radius: 16px;
    border: 1px solid __CARD_BORDER__;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08);
    padding: 20px;
    margin-bottom: 16px;
    transition: all 0.2s ease;
}
.card:hover { box-shadow: 0 12px 24px rgba(15, 23, 42, 0.1); }
.card,
.card p,
.card span,
.card label,
.card div,
.card h1,
.card h2,
.card h3,
.card h4,
.card h5,
.card h6 {
    color: __TEXT_COLOR__ !important;
}
.primary-btn button {
    background: __BUTTON_BG__ !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}
.stButton > button {
    background: __BUTTON_BG__ !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}
.stDownloadButton > button {
    background: __BUTTON_BG__ !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    cursor: pointer !important;
}
.stDownloadButton > button:hover {
    opacity: 0.9 !important;
    cursor: pointer !important;
}
.xai-controls label,
.xai-controls span,
.xai-controls p {
    color: __TEXT_COLOR__ !important;
}
.stCheckbox label,
.stSlider label,
.stSelectbox label,
.stMarkdown,
.stCaption,
.stRadio label {
    color: __TEXT_COLOR__ !important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] strong {
    color: __TEXT_COLOR__ !important;
}
section[data-testid="stFileUploader"] {
    border: 2px dashed #3b82f6;
    border-radius: 16px;
    background: __UPLOADER_BG__;
    padding: 14px;
}
section[data-testid="stFileUploader"] * {
    color: __TEXT_COLOR__ !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    background: __UPLOADER_BG__ !important;
    border: 1px dashed #60a5fa !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: __TEXT_COLOR__ !important;
}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
    background: __BUTTON_BG__ !important;
    color: #ffffff !important;
    border: none !important;
}
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] p,
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] span {
    color: #ffffff !important;
}
[data-testid="stFileUploaderFileName"] {
    color: __TEXT_COLOR__ !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] *,
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] div,
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] span {
    color: __TEXT_COLOR__ !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] label {
    color: __TEXT_COLOR__ !important;
}
.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div > input,
div[data-baseweb="textarea"] textarea {
    color: __INPUT_TEXT__ !important;
    background: __INPUT_BG__ !important;
}
.stSelectbox div[data-baseweb="select"] span,
.stSelectbox div[data-baseweb="select"] svg {
    color: __INPUT_TEXT__ !important;
    fill: __INPUT_TEXT__ !important;
}
.stMultiSelect div[data-baseweb="tag"] {
    color: __INPUT_TEXT__ !important;
}
.xai-controls div[data-testid="stToggle"] label,
.xai-controls div[data-testid="stToggle"] span,
.xai-controls div[data-testid="stToggle"] p {
    color: __TEXT_COLOR__ !important;
}
.xai-controls div[data-testid="stToggle"] {
    border: 1px solid __CARD_BORDER__;
    border-radius: 12px;
    padding: 8px 12px;
    background: __TOGGLE_BG__;
}
.xai-controls div[data-testid="stToggle"] input[type="checkbox"] {
    accent-color: __BUTTON_BG__ !important;
}
.xai-controls div[data-testid="stToggle"] [role="switch"] {
    border: 1px solid __CARD_BORDER__ !important;
    background: #cbd5e1 !important;
}
div[data-testid="stAlert"] {
    border: 1px solid __CARD_BORDER__ !important;
    background: __ALERT_BG__ !important;
}
div[data-testid="stAlert"] * {
    color: __TEXT_COLOR__ !important;
}
.status-pill {
    display: inline-block;
    border-radius: 999px;
    padding: 4px 12px;
    font-weight: 700;
    border: 1px solid __CARD_BORDER__;
    background: __STATUS_BG__;
    color: __STATUS_TEXT__ !important;
}
@media (max-width: 980px) {
    .dashboard-shell { display: block; }
    .control-panel, .content-panel { width: 100%; max-height: none; position: static; }
}
@media (max-width: 768px) {
    .block-container {
        width: 100%;
        padding: 12px 10px;
    }
    .card {
        padding: 14px;
    }
    .stButton > button,
    .stDownloadButton > button {
        width: 100% !important;
        min-height: 44px !important;
        font-size: 14px !important;
    }
    .stImage img {
        width: 100% !important;
        height: auto !important;
    }
}
</style>
    """
    css = (
        css.replace("__APP_BG__", app_bg)
        .replace("__TEXT_COLOR__", text_color)
        .replace("__CARD_BG__", card_bg)
        .replace("__CARD_BORDER__", card_border)
        .replace("__BUTTON_BG__", button_bg)
        .replace("__INPUT_BG__", input_bg)
        .replace("__INPUT_TEXT__", input_text)
        .replace("__UPLOADER_BG__", uploader_bg)
        .replace("__TOGGLE_BG__", toggle_bg)
        .replace("__ALERT_BG__", alert_bg)
        .replace("__STATUS_BG__", status_bg)
        .replace("__STATUS_TEXT__", status_text)
        .replace("__HEADER_BG__", header_bg)
        .replace("__HEADER_TEXT__", header_text)
    )
    st.markdown(css, unsafe_allow_html=True)


def init_state():
    st.session_state.setdefault("upload_results", {})
    st.session_state.setdefault("upload_token", None)
    st.session_state.setdefault("upload_name", "")
    st.session_state.setdefault("upload_bytes", None)
    st.session_state.setdefault("camera_on", False)
    st.session_state.setdefault("camera_exp", None)
    st.session_state.setdefault("history_local", [])
    st.session_state.setdefault("stress_result", None)
    st.session_state.setdefault("uncertainty_result", None)
    st.session_state.setdefault("counterfactual", None)
    st.session_state.setdefault("streamlit_dark_mode", False)


def render_header():
    st.markdown(
        """
<div class="hero">
  <h2 style="margin:0;">Vision Transformer Image Recognition with Explainable AI</h2>
  <p style="margin:8px 0 0 0;">
    Full-width, modern, interactive AI dashboard with robustness stress testing, uncertainty analysis,
    contrastive reasoning, and professional report export.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_controls_panel(in_sidebar: bool = False) -> dict:
    if in_sidebar:
        st.markdown('<div class="xai-controls sidebar-controls">', unsafe_allow_html=True)
        st.markdown("## XAI Controls")
    else:
        st.markdown('<div class="card xai-controls">', unsafe_allow_html=True)
        st.markdown("### XAI Controls")
    st.toggle("🌙 Dark Mode", key="streamlit_dark_mode")
    st.markdown("---")
    show_explanation = st.checkbox("Show Explanation", value=True)
    show_contrastive = st.checkbox("Show Contrastive Explanation", value=True)
    show_attention = st.checkbox("Show Attention Maps", value=True)
    show_gradcam = st.checkbox("Show Grad-CAM", value=True)
    show_ig = st.checkbox("Show Integrated Gradients", value=False)
    st.markdown("---")
    frame_skip = st.slider("Webcam Frame Skip", 2, 20, 10, 1)
    interval_sec = st.slider("Webcam Interval (sec)", 1.0, 4.0, 2.5, 0.5)
    with LIVE_STATE.lock:
        LIVE_STATE.frame_skip = frame_skip
        LIVE_STATE.interval_sec = interval_sec
    st.markdown("---")
    st.markdown("### Stress Test Controls")
    blur_value = st.slider("Blur", 0, 17, 0, 1)
    noise_value = st.slider("Noise (Gaussian)", 0.0, 0.4, 0.0, 0.01)
    mask_value = st.slider("Patch Mask Size", 0, 160, 0, 4)
    st.markdown("</div>", unsafe_allow_html=True)

    return {
        "show_explanation": show_explanation,
        "show_contrastive": show_contrastive,
        "show_attention": show_attention,
        "show_gradcam": show_gradcam,
        "show_ig": show_ig,
        "stress": StressConfig(blur_kernel=blur_value, noise_level=noise_value, mask_size=mask_value),
    }


def parse_uploaded_image(uploaded) -> tuple[Image.Image | None, str]:
    if uploaded is None:
        return None, ""
    try:
        return Image.open(uploaded).convert("RGB"), ""
    except (UnidentifiedImageError, OSError, ValueError):
        return None, "Invalid image file. Please upload a valid PNG/JPG/JPEG image."


def confidence_to_pct(confidence: float) -> float:
    return confidence * 100.0 if confidence <= 1.0 else confidence


def _query_value(name: str) -> str:
    value = st.query_params.get(name, "")
    if isinstance(value, list):
        return str(value[0]).strip()
    return str(value).strip()


def get_active_user_key() -> str:
    email = _query_value("email").lower()
    username = _query_value("user").lower()
    return email or username or "anonymous_streamlit"


def encode_image_to_b64(image: Image.Image, max_size: int = 512) -> str:
    thumb = image.copy()
    thumb.thumbnail((max_size, max_size))
    buff = io.BytesIO()
    thumb.save(buff, format="JPEG", quality=85)
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def estimate_focus_region(image_arr) -> str:
    if image_arr is None:
        return "unknown region"
    arr = image_arr.astype("float32")
    gray = arr.mean(axis=2)
    ys, xs = (gray >= np.percentile(gray, 85)).nonzero()
    if len(xs) == 0:
        return "diffuse region"
    h, w = gray.shape
    x_mean = xs.mean() / w
    y_mean = ys.mean() / h
    horizontal = "left" if x_mean < 0.33 else ("right" if x_mean > 0.66 else "center")
    vertical = "upper" if y_mean < 0.33 else ("lower" if y_mean > 0.66 else "middle")
    return f"{vertical}-{horizontal}"


def persist_history(image: Image.Image, image_name: str, mode: str, result: dict) -> str | None:
    user_key = get_active_user_key()
    record = {
        "user_key": user_key,
        "image_name": image_name,
        "image_b64": encode_image_to_b64(image),
        "model_mode": mode,
        "model_name": result.get("model_name"),
        "prediction": result.get("label"),
        "confidence": float(confidence_to_pct(result.get("confidence", 0.0))),
        "top_predictions": result.get("top_predictions", []),
        "technical_explanation": result.get("technical_explanation"),
        "llm_explanation": result.get("explanation"),
        "contrastive_explanation": result.get("contrastive_explanation"),
    }
    if MONGO_READY:
        return save_prediction_history(record)
    local = st.session_state["history_local"]
    local_id = f"local_{len(local)+1}"
    local.insert(0, {**record, "_id": local_id})
    st.session_state["history_local"] = local[:12]
    return local_id


def delete_history_item(record_id: str) -> bool:
    user_key = get_active_user_key()
    if MONGO_READY:
        return bool(delete_prediction_history(record_id, user_key=user_key))
    local = st.session_state.get("history_local", [])
    before = len(local)
    local = [
        row
        for row in local
        if not (str(row.get("_id")) == str(record_id) and str(row.get("user_key", "")) == str(user_key))
    ]
    st.session_state["history_local"] = local
    return len(local) < before


def run_explanations(
    image: Image.Image,
    selected_modes: list[str],
    settings: dict,
    image_name: str,
    progress_callback=None,
) -> dict[str, dict]:
    results = {}
    total = max(1, len(selected_modes))
    for idx, mode in enumerate(selected_modes, start=1):
        if progress_callback:
            progress_callback(current=idx - 1, total=total, mode=mode)
        exp = explain_image(
            image=image,
            model_mode=mode,
            include_gradcam=settings["show_gradcam"],
            include_attention_map=settings["show_attention"],
            include_rollout=settings["show_attention"],
            include_llm_explanation=settings["show_explanation"],
            include_integrated_gradients=settings["show_ig"],
            resize_224=False,
        )
        exp["history_id"] = persist_history(image, image_name, mode, exp)
        results[mode] = exp
        if progress_callback:
            progress_callback(current=idx, total=total, mode=mode)
    return results


def render_topk_chart(top_predictions: list[dict], chart_key: str):
    if not top_predictions:
        st.info("No top-k data.")
        return
    labels = [x["label"] for x in top_predictions][::-1]
    values = [confidence_to_pct(x["confidence"]) for x in top_predictions][::-1]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.barh(labels, values, color=["#3b82f6", "#2563eb", "#1e40af"][: len(labels)])
    ax.set_xlabel("Confidence (%)")
    ax.grid(axis="x", linestyle="--", alpha=0.2)
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)


def render_upload_prediction(settings: dict) -> tuple[Image.Image | None, dict[str, dict]]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## Upload")
    left, right = st.columns([0.7, 0.3], vertical_alignment="top")
    with left:
        uploaded = st.file_uploader(
            "Upload image",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key="image_uploader",
        )
    with right:
        compare = st.toggle("Compare Models", value=False)
        if compare:
            selected_modes = st.multiselect("Model Modes", ["pretrained", "custom"], default=["pretrained", "custom"])
        else:
            selected_modes = [st.selectbox("Model Mode", ["pretrained", "custom"], index=0)]
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        predict_clicked = st.button("Predict", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    image = None
    image_name = st.session_state.get("upload_name", "uploaded_image")
    image_bytes = None
    err = ""

    if uploaded is not None:
        image, err = parse_uploaded_image(uploaded)
        if err:
            st.error(err)
            return None, st.session_state.get("upload_results", {})
        image_bytes = uploaded.getvalue()
        image_name = uploaded.name
        st.session_state["upload_bytes"] = image_bytes
        st.session_state["upload_name"] = image_name
    else:
        cached_bytes = st.session_state.get("upload_bytes")
        if cached_bytes:
            try:
                image = Image.open(io.BytesIO(cached_bytes)).convert("RGB")
                image_bytes = cached_bytes
            except (UnidentifiedImageError, OSError, ValueError):
                st.session_state["upload_bytes"] = None
                st.session_state["upload_name"] = ""
                st.warning("Please upload an image.")
                return None, st.session_state.get("upload_results", {})
        else:
            st.warning("Please upload an image.")
            return None, st.session_state.get("upload_results", {})

    token = hashlib.sha1(image_bytes).hexdigest()
    if st.session_state.get("upload_token") != token:
        st.session_state["upload_token"] = token
        st.session_state["upload_results"] = {}
        st.session_state["upload_name"] = image_name

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## Prediction")
    st.image(image, width="stretch")
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    if predict_clicked:
        if not selected_modes:
            st.warning("Select at least one model.")
        else:
            progress_bar = progress_placeholder.progress(0)

            def _on_progress(current: int, total: int, mode: str):
                percent = int((current / max(total, 1)) * 100)
                progress_bar.progress(percent)
                if current < total:
                    status_placeholder.info(f"Processing `{mode}` model... ({current + 1}/{total})")
                else:
                    status_placeholder.success("Processing completed. Results are ready.")

            st.session_state["upload_results"] = run_explanations(
                image=image,
                selected_modes=selected_modes,
                settings=settings,
                image_name=image_name,
                progress_callback=_on_progress,
            )
            progress_bar.progress(100)
    results = st.session_state.get("upload_results", {})
    if results:
        cols = st.columns(len(results))
        for col, (mode, result) in zip(cols, results.items()):
            with col:
                st.markdown(
                    f"""
<div class="card" style="margin:0;">
<h4 style="margin-top:0;">{mode.title()}</h4>
<p><b>Prediction:</b> {result['label']}</p>
<p><b>Confidence:</b> {confidence_to_pct(result['confidence']):.2f}%</p>
<p><b>Device:</b> {result.get('device')}</p>
</div>
                    """,
                    unsafe_allow_html=True,
                )
                top_df = pd.DataFrame(
                    [{"Class": t["label"], "Confidence (%)": round(confidence_to_pct(t["confidence"]), 2)} for t in result["top_predictions"]]
                )
                st.dataframe(top_df, hide_index=True, use_container_width=True)
                render_topk_chart(result["top_predictions"], chart_key=f"topk_{mode}")
    else:
        st.info("Click Predict to generate outputs.")
    st.markdown("</div>", unsafe_allow_html=True)
    return image, results


def render_explainability(image: Image.Image, results: dict[str, dict], settings: dict):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## Explainability")
    for mode, result in results.items():
        st.markdown(f"### {mode.title()} Model")
        if settings["show_explanation"]:
            st.write(result.get("technical_explanation") or result.get("explanation"))
        if settings["show_contrastive"]:
            st.info(result.get("contrastive_explanation"))

        cols = st.columns(5)
        with cols[0]:
            st.caption("Original")
            st.image(image, width="stretch")
        with cols[1]:
            st.caption("Grad-CAM")
            if settings["show_gradcam"] and result.get("gradcam_overlay") is not None:
                st.image(result["gradcam_overlay"], width="stretch")
            else:
                st.info("Disabled")
        with cols[2]:
            st.caption("Attention Map")
            if settings["show_attention"] and result.get("attention_map_overlay") is not None:
                st.image(result["attention_map_overlay"], width="stretch")
            else:
                st.info("Disabled")
        with cols[3]:
            st.caption("Attention Rollout")
            if settings["show_attention"] and result.get("attention_rollout_overlay") is not None:
                st.image(result["attention_rollout_overlay"], width="stretch")
            else:
                st.info("Disabled")
        with cols[4]:
            st.caption("Integrated Gradients")
            if settings["show_ig"] and result.get("integrated_gradients_overlay") is not None:
                st.image(result["integrated_gradients_overlay"], width="stretch")
            else:
                st.info("Disabled")
    st.markdown("</div>", unsafe_allow_html=True)


def render_stress_test(image: Image.Image, results: dict[str, dict], settings: dict):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## Stress Test")
    st.caption("Interactive robustness testing with blur, noise, and patch masking.")
    if st.button("Run Stress Test"):
        if not results:
            st.warning("Run prediction first.")
        else:
            mode = next(iter(results.keys()))
            stressed = apply_stress_transform(image, settings["stress"])
            with st.spinner("Evaluating stressed image..."):
                stressed_result = explain_image(
                    image=stressed,
                    model_mode=mode,
                    include_gradcam=settings["show_gradcam"],
                    include_attention_map=settings["show_attention"],
                    include_rollout=settings["show_attention"],
                    include_llm_explanation=settings["show_explanation"],
                    include_integrated_gradients=settings["show_ig"],
                    resize_224=False,
                )
            st.session_state["stress_result"] = {"image": stressed, "result": stressed_result, "mode": mode}

    stress_payload = st.session_state.get("stress_result")
    if stress_payload:
        stressed = stress_payload["image"]
        stressed_result = stress_payload["result"]
        mode = stress_payload["mode"]
        original = results.get(mode)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Original")
            st.image(image, width="stretch")
        with c2:
            st.caption("Modified")
            st.image(stressed, width="stretch")
        if original:
            delta = confidence_to_pct(stressed_result["confidence"] - original["confidence"])
            orig_focus = estimate_focus_region(original.get("gradcam_overlay"))
            stress_focus = estimate_focus_region(stressed_result.get("gradcam_overlay"))
            st.write(f"Prediction change: `{original['label']}` → `{stressed_result['label']}`")
            st.write(f"Confidence difference: `{delta:+.2f}%`")
            st.info(summarize_focus_shift(orig_focus, stress_focus))
    st.markdown("</div>", unsafe_allow_html=True)


def render_counterfactual(image: Image.Image, results: dict[str, dict]):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## Why Not Another Class?")
    if st.button("Generate Counterfactual Explanation"):
        if not results:
            st.warning("Run prediction first.")
        else:
            mode = next(iter(results.keys()))
            pred = predict_image(image, model_mode=mode, resize_224=False, return_attentions=False)
            top = pred.get("top_predictions", [])
            if len(top) < 2:
                st.warning("Need at least top-2 classes.")
            else:
                top1, top2 = top[0], top[1]
                with st.spinner("Computing class-wise Grad-CAM..."):
                    cam_top1 = generate_gradcam_for_class(image=image, model_mode=mode, class_idx=top1["pred_idx"])
                    cam_top2 = generate_gradcam_for_class(image=image, model_mode=mode, class_idx=top2["pred_idx"])
                st.session_state["counterfactual"] = {
                    "top1": top1,
                    "top2": top2,
                    "cam_top1": cam_top1,
                    "cam_top2": cam_top2,
                }

    cf = st.session_state.get("counterfactual")
    if cf:
        c1, c2 = st.columns(2)
        with c1:
            st.caption(f"Why this class: {cf['top1']['label']}")
            st.image(cf["cam_top1"], width="stretch")
        with c2:
            st.caption(f"Why not {cf['top2']['label']}")
            st.image(cf["cam_top2"], width="stretch")
        st.write(
            f"The model prioritizes discriminative features for `{cf['top1']['label']}` "
            f"over `{cf['top2']['label']}` based on stronger regional activation and feature attribution."
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_uncertainty(image: Image.Image, results: dict[str, dict]):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## AI Confidence Scorecard")
    if st.button("Run Stability Analyzer"):
        if not results:
            st.warning("Run prediction first.")
        else:
            mode = next(iter(results.keys()))
            with st.spinner("Running uncertainty analysis..."):
                stability = analyze_prediction_stability(image=image, predict_fn=predict_image, model_mode=mode, runs=5)
            st.session_state["uncertainty_result"] = stability

    stability = st.session_state.get("uncertainty_result")
    if stability:
        st.markdown(f"**Badge:** `{stability.badge}`")
        st.progress(min(max(stability.consistency, 0.0), 1.0))
        st.write(f"Mean confidence: {confidence_to_pct(stability.mean_confidence):.2f}%")
        st.write(f"Std deviation: {confidence_to_pct(stability.std_confidence):.2f}%")
        st.write(f"Prediction consistency: {stability.consistency * 100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)


def render_report_section(image: Image.Image, results: dict[str, dict]):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## Report")
    if not results:
        st.info("Run prediction first to enable report export.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    mode = st.selectbox("Report Model", list(results.keys()), key="report_mode")
    result = results[mode]
    report_payload = {
        "image": image,
        "label": result.get("label"),
        "confidence": confidence_to_pct(result.get("confidence", 0.0)),
        "model_name": result.get("model_name"),
        "model_mode": result.get("model_mode"),
        "technical_explanation": result.get("technical_explanation"),
        "explanation": result.get("explanation"),
        "gradcam_overlay": result.get("gradcam_overlay"),
        "attention_map_overlay": result.get("attention_map_overlay"),
        "attention_rollout_overlay": result.get("attention_rollout_overlay"),
        "integrated_gradients_overlay": result.get("integrated_gradients_overlay"),
    }
    pdf_bytes, filename = build_pdf_report(report_payload)
    if pdf_bytes is None:
        st.warning("PDF dependency missing. Install `fpdf2` to enable report export.")
    else:
        st.download_button("Download AI Report", data=pdf_bytes, file_name=filename, mime="application/pdf")
    st.markdown("</div>", unsafe_allow_html=True)


def render_history_gallery():
    user_key = get_active_user_key()
    if MONGO_READY:
        rows = fetch_recent_history(limit=12, user_key=user_key)
    else:
        rows = [row for row in st.session_state.get("history_local", []) if str(row.get("user_key", "")) == str(user_key)]
    if not rows:
        return
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## History Gallery")
    cols = st.columns(3)
    for idx, row in enumerate(rows):
        with cols[idx % 3]:
            record_id = str(row.get("_id", f"row_{idx}"))
            st.markdown(
                f"""
<div class="card" style="margin:0 0 10px 0; padding:14px;">
<p><b>{row.get('image_name', '--')}</b></p>
<p>Prediction: {row.get('prediction', '--')}</p>
<p>Confidence: {row.get('confidence', '--')}%</p>
<p>Model: {row.get('model_mode', '--')}</p>
</div>
                """,
                unsafe_allow_html=True,
            )
            image_b64 = row.get("image_b64")
            if image_b64:
                try:
                    image_bytes = base64.b64decode(image_b64)
                    st.image(Image.open(io.BytesIO(image_bytes)), width="stretch")
                except Exception:
                    pass
            if st.button("Delete", key=f"delete_history_{record_id}", use_container_width=True):
                deleted = delete_history_item(record_id)
                if deleted:
                    st.success("History item deleted.")
                else:
                    st.warning("Could not delete this history item.")
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_live_camera(settings: dict):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## Live Detection")
    if not st.session_state.get("camera_on", False):
        with LIVE_STATE.lock:
            LIVE_STATE.status = "OFF"
            LIVE_STATE.label = "--"
            LIVE_STATE.confidence = 0.0
            LIVE_STATE.last_frame_rgb = None
            LIVE_STATE.device = "--"
            LIVE_STATE.fp16_enabled = False
            LIVE_STATE.top_predictions = None
    c1, c2, c3 = st.columns([0.2, 0.2, 0.6], vertical_alignment="center")
    with c1:
        if st.button("Start Camera", use_container_width=True):
            st.session_state["camera_on"] = True
            with LIVE_STATE.lock:
                LIVE_STATE.status = "ON"
    with c2:
        if st.button("Stop Camera", use_container_width=True):
            st.session_state["camera_on"] = False
            with LIVE_STATE.lock:
                LIVE_STATE.status = "OFF"
                LIVE_STATE.label = "--"
                LIVE_STATE.confidence = 0.0
                LIVE_STATE.last_frame_rgb = None
                LIVE_STATE.device = "--"
                LIVE_STATE.fp16_enabled = False
                LIVE_STATE.top_predictions = None
    with c3:
        mode = st.selectbox("Camera Model", ["pretrained", "custom"], index=0, key="camera_mode")
        with LIVE_STATE.lock:
            LIVE_STATE.model_mode = mode
        st.markdown(f"<span class='status-pill'>Camera Status: {LIVE_STATE.status}</span>", unsafe_allow_html=True)
        st.caption("Adaptive runtime: uses GPU when available, otherwise CPU, with low-frequency frame processing.")

    if st.session_state.get("camera_on", False):
        if STREAMLIT_WEBRTC_AVAILABLE and WebcamPredictor is not None:
            ctx = webrtc_streamer(
                key="vision-live-cam",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=WebcamPredictor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            if not ctx.state.playing:
                st.warning("Waiting for camera permission...")
        else:
            st.warning(
                "Camera streaming is unavailable because streamlit-webrtc is not installed or supported in this environment."
            )
            st.info("Use app/streamlit_app_simple.py for Streamlit Cloud deployment without camera support.")
    else:
        st.info("Camera is OFF. No prediction is performed.")

    with LIVE_STATE.lock:
        current_label = LIVE_STATE.label
        current_conf = LIVE_STATE.confidence
        current_device = LIVE_STATE.device
        current_fp16 = LIVE_STATE.fp16_enabled
        current_topk = list(LIVE_STATE.top_predictions or [])

    st.write(f"Prediction: **{current_label}** | Confidence: **{confidence_to_pct(current_conf):.2f}%**")
    st.caption(f"Device: {current_device} | FP16: {current_fp16}")
    if current_topk:
        top_df = pd.DataFrame(
            [{"Class": t["label"], "Confidence (%)": round(confidence_to_pct(t["confidence"]), 2)} for t in current_topk]
        )
        st.dataframe(top_df, hide_index=True, use_container_width=True)

    explain_clicked = st.button("Explain Current Camera Frame", disabled=not st.session_state.get("camera_on", False))
    if explain_clicked:
        with LIVE_STATE.lock:
            frame = LIVE_STATE.last_frame_rgb.copy() if LIVE_STATE.last_frame_rgb is not None else None
            mode = LIVE_STATE.model_mode
        if frame is None:
            st.warning("No frame captured yet.")
        else:
            with st.spinner("Generating camera explanation..."):
                st.session_state["camera_exp"] = explain_image(
                    image=Image.fromarray(frame),
                    model_mode=mode,
                    include_gradcam=settings["show_gradcam"],
                    include_attention_map=settings["show_attention"],
                    include_rollout=settings["show_attention"],
                    include_llm_explanation=settings["show_explanation"],
                    include_integrated_gradients=settings["show_ig"],
                    resize_224=True,
                )
    exp = st.session_state.get("camera_exp")
    if exp:
        st.markdown("### Camera Explainability")
        if exp.get("technical_explanation"):
            st.write(exp.get("technical_explanation"))
        if exp.get("explanation"):
            st.info(exp.get("explanation"))
        if exp.get("contrastive_explanation"):
            st.caption(exp.get("contrastive_explanation"))

        exp_topk = exp.get("top_predictions") or []
        if exp_topk:
            exp_top_df = pd.DataFrame(
                [{"Class": t["label"], "Confidence (%)": round(confidence_to_pct(t["confidence"]), 2)} for t in exp_topk]
            )
            st.dataframe(exp_top_df, hide_index=True, use_container_width=True)

        visual_items = [
            ("Grad-CAM", exp.get("gradcam_overlay")),
            ("Attention Map", exp.get("attention_map_overlay")),
            ("Attention Rollout", exp.get("attention_rollout_overlay")),
            ("Integrated Gradients", exp.get("integrated_gradients_overlay")),
        ]
        available_visuals = [(name, image) for name, image in visual_items if image is not None]
        if available_visuals:
            vis_cols = st.columns(len(available_visuals))
            for col, (title, overlay) in zip(vis_cols, available_visuals):
                with col:
                    st.caption(title)
                    st.image(overlay, width="stretch")
        else:
            st.info("No explainability visualization is enabled.")
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    init_state()
    set_styles()
    render_header()
    with st.sidebar:
        settings = render_controls_panel(in_sidebar=True)

    image, results = render_upload_prediction(settings)
    if image is not None and results:
        render_explainability(image, results, settings)
        render_stress_test(image, results, settings)
        render_counterfactual(image, results)
        render_uncertainty(image, results)
        render_report_section(image, results)
    render_history_gallery()
    render_live_camera(settings)


if __name__ == "__main__":
    main()
