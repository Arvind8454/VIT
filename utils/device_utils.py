from __future__ import annotations

from contextlib import nullcontext

import torch


def get_device() -> torch.device:
    """Return CUDA when available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_cuda(device: torch.device | str) -> bool:
    name = str(device)
    return name.startswith("cuda")


def use_fp16(device: torch.device | str) -> bool:
    """
    Mixed precision is enabled only on CUDA.
    Keep CPU in float32 for stability.
    """
    return is_cuda(device)


def autocast_context(device: torch.device | str, enabled: bool = True):
    """Return an autocast context for safe mixed-precision inference."""
    if enabled and is_cuda(device):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def maybe_half_tensor(tensor: torch.Tensor, device: torch.device | str, enabled: bool = True) -> torch.Tensor:
    """Cast tensor to fp16 only when safe and requested."""
    if enabled and use_fp16(device):
        return tensor.half()
    return tensor.float()


def clear_device_cache(device: torch.device | str) -> None:
    """Release CUDA cache when needed to reduce memory pressure."""
    if is_cuda(device):
        torch.cuda.empty_cache()
