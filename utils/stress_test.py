from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class StressConfig:
    blur_kernel: int = 0
    noise_level: float = 0.0
    mask_size: int = 0


def _normalize_kernel(value: int) -> int:
    if value <= 0:
        return 0
    # Gaussian blur requires odd kernel size.
    return value if value % 2 == 1 else value + 1


def apply_stress_transform(image: Image.Image, config: StressConfig) -> Image.Image:
    """Apply blur, Gaussian noise, and patch masking to the input image."""
    arr = np.array(image).astype(np.float32)

    kernel = _normalize_kernel(config.blur_kernel)
    if kernel > 1:
        arr = cv2.GaussianBlur(arr, (kernel, kernel), 0)

    if config.noise_level > 0:
        sigma = config.noise_level * 255.0
        noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
        arr = arr + noise

    if config.mask_size > 0:
        h, w, _ = arr.shape
        size = min(config.mask_size, h, w)
        y1 = max(0, (h // 2) - (size // 2))
        x1 = max(0, (w // 2) - (size // 2))
        arr[y1 : y1 + size, x1 : x1 + size, :] = 0

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def summarize_focus_shift(original_focus: str, stressed_focus: str) -> str:
    return (
        f"The model focus shifted from {original_focus} to {stressed_focus}, "
        "indicating sensitivity to perturbations in visual evidence."
    )
