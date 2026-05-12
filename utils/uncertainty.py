from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance


@dataclass
class StabilityResult:
    mean_confidence: float
    std_confidence: float
    consistency: float
    badge: str
    predictions: list[str]
    confidences: list[float]


def _random_transform(image: Image.Image) -> Image.Image:
    img = image.copy()
    w, h = img.size

    # Random crop with mild scale change.
    scale = np.random.uniform(0.92, 1.0)
    nw, nh = int(w * scale), int(h * scale)
    x1 = np.random.randint(0, max(1, w - nw + 1))
    y1 = np.random.randint(0, max(1, h - nh + 1))
    img = img.crop((x1, y1, x1 + nw, y1 + nh)).resize((w, h))

    # Brightness jitter.
    brightness_factor = np.random.uniform(0.9, 1.1)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # Slight rotation.
    angle = float(np.random.uniform(-8, 8))
    img = img.rotate(angle, resample=Image.BILINEAR)
    return img


def analyze_prediction_stability(
    image: Image.Image,
    predict_fn,
    model_mode: str,
    runs: int = 5,
) -> StabilityResult:
    """
    Estimate uncertainty by repeated stochastic augmentations.
    predict_fn signature: predict_fn(image, model_mode=..., resize_224=..., return_attentions=...)
    """
    runs = max(3, min(runs, 8))
    labels: list[str] = []
    confidences: list[float] = []

    for _ in range(runs):
        aug = _random_transform(image)
        pred = predict_fn(aug, model_mode=model_mode, resize_224=False, return_attentions=False)
        labels.append(pred["label"])
        confidences.append(float(pred["confidence"]))

    conf_arr = np.array(confidences, dtype=np.float32)
    mean_conf = float(conf_arr.mean())
    std_conf = float(conf_arr.std())

    most_common = max(set(labels), key=labels.count)
    consistency = labels.count(most_common) / len(labels)

    if consistency >= 0.8 and std_conf <= 0.08:
        badge = "High Stability"
    elif consistency >= 0.6 and std_conf <= 0.15:
        badge = "Moderate Stability"
    else:
        badge = "Low Stability"

    return StabilityResult(
        mean_confidence=mean_conf,
        std_confidence=std_conf,
        consistency=float(consistency),
        badge=badge,
        predictions=labels,
        confidences=confidences,
    )
