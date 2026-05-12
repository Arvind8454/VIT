from __future__ import annotations

import argparse
import os
import sys

from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.prediction_service import explain_image


def load_class_names(model_dir: str, fallback: list[str]):
    path = os.path.join(model_dir, "class_names.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        if names:
            return names
    return fallback


def predict_with_explanations(_model, _processor, image: Image.Image, _device: str, use_tta: bool = True):
    # Kept for compatibility with existing Streamlit imports.
    exp = explain_image(
        image,
        model_mode="pretrained",
        include_gradcam=True,
        include_attention_map=True,
        include_rollout=True,
        resize_224=False,
    )
    # Return legacy shape + additional overlays for upgraded UI.
    return (
        exp["pred_idx"],
        exp["confidence"],
        exp["gradcam_overlay"],
        exp["attention_rollout_overlay"],
        exp["attention_map_overlay"],
    )


def main():
    parser = argparse.ArgumentParser(description="ViT inference with explainability")
    parser.add_argument("--image_path", required=True)
    args = parser.parse_args()

    image = Image.open(args.image_path).convert("RGB")
    exp = explain_image(
        image,
        model_mode="pretrained",
        include_gradcam=True,
        include_attention_map=True,
        include_rollout=True,
        resize_224=False,
    )
    print(f"Prediction: {exp['label']} | Confidence: {exp['confidence']:.4f}")

    if exp["gradcam_overlay"] is not None:
        Image.fromarray(exp["gradcam_overlay"]).save("gradcam_overlay.png")
    if exp["attention_map_overlay"] is not None:
        Image.fromarray(exp["attention_map_overlay"]).save("attention_map_overlay.png")
    if exp["attention_rollout_overlay"] is not None:
        Image.fromarray(exp["attention_rollout_overlay"]).save("attention_rollout_overlay.png")
    print("Saved explainability outputs.")


if __name__ == "__main__":
    main()
