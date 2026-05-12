# Streamlit Cloud Compatible Version
# Simplified to avoid OpenCV/webcam issues

from __future__ import annotations

import base64
import hashlib
import io
import os
import sys
import time
from dataclasses import dataclass

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

# Page configuration
st.set_page_config(
    page_title="ViT Image Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class PredictionResult:
    """Data class for prediction results"""
    class_name: str
    confidence: float
    all_predictions: dict

def load_image() -> Image.Image | None:
    """Load image from file upload"""
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload an image for classification"
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except UnidentifiedImageError:
            st.error("❌ Invalid image file. Please upload a valid image.")
            return None
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
            return None
    return None

def display_predictions(predictions: dict, title: str = "Predictions"):
    """Display prediction results in a nice format"""
    st.subheader(f"🎯 {title}")

    # Sort predictions by confidence
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    # Display top prediction prominently
    top_class, top_conf = sorted_preds[0]
    st.markdown(f"""
    <div class="prediction-box">
        <h3 style="color: #1f77b4; margin: 0;">Top Prediction: {top_class}</h3>
        <p style="font-size: 1.2rem; margin: 0.5rem 0;">Confidence: {top_conf:.1f}%</p>
        <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px;">
            <div style="background-color: #1f77b4; width: {top_conf}%; height: 20px; border-radius: 10px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Display all predictions
    st.subheader("📊 All Predictions")
    for class_name, confidence in sorted_preds[:5]:  # Show top 5
        st.write(f"**{class_name}**: {confidence:.1f}%")

def display_explanations(explanations: dict):
    """Display XAI explanations"""
    st.subheader("🔍 Explainability Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        if explanations.get("gradcam_overlay") is not None:
            st.write("**Grad-CAM**")
            st.image(explanations["gradcam_overlay"], use_column_width=True)
        else:
            st.write("Grad-CAM: Not available")

    with col2:
        if explanations.get("attention_map_overlay") is not None:
            st.write("**Attention Map**")
            st.image(explanations["attention_map_overlay"], use_column_width=True)
        else:
            st.write("Attention Map: Not available")

    with col3:
        if explanations.get("integrated_gradients_overlay") is not None:
            st.write("**Integrated Gradients**")
            st.image(explanations["integrated_gradients_overlay"], use_column_width=True)
        else:
            st.write("Integrated Gradients: Not available")

def main():
    """Main Streamlit application"""
    st.markdown('<h1 class="main-header">🖼️ ViT Image Classifier</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        enable_explanations = st.checkbox("Enable XAI Explanations", value=True)
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This app uses Vision Transformer (ViT) for image classification with explainable AI techniques.")

    # Initialize model
    try:
        with st.spinner("Loading ViT model..."):
            preload_pretrained_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        image = load_image()

        if image is not None:
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Classification button
            if st.button("🔍 Classify Image", type="primary", use_container_width=True):
                with st.spinner("Classifying image..."):
                    try:
                        # Get predictions
                        predictions = predict_image(image)

                        # Store results
                        result = PredictionResult(
                            class_name=max(predictions, key=predictions.get),
                            confidence=max(predictions.values()),
                            all_predictions=predictions
                        )

                        # Store in session state
                        st.session_state.last_result = result
                        st.session_state.last_image = image

                    except Exception as e:
                        st.error(f"❌ Classification failed: {str(e)}")

    with col2:
        st.subheader("🎯 Results")

        if 'last_result' in st.session_state:
            result = st.session_state.last_result
            display_predictions(result.all_predictions)

            # Show explanations if enabled
            if enable_explanations and 'last_image' in st.session_state:
                try:
                    with st.spinner("Generating explanations..."):
                        explanations = explain_image(st.session_state.last_image)
                        display_explanations(explanations)
                except Exception as e:
                    st.warning(f"⚠️ Could not generate explanations: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and Vision Transformers")

if __name__ == "__main__":
    main()
