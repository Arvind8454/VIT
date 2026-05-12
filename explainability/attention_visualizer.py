from __future__ import annotations

import os

import cv2
import numpy as np
import torch


def _normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    return arr


def _debug(msg: str):
    if os.getenv("ATTENTION_DEBUG", "0") == "1":
        print(msg, flush=True)


def attention_rollout(attentions, discard_ratio: float = 0.0):
    if attentions is None or len(attentions) == 0:
        return None
    # attentions: list of [B, heads, tokens, tokens]
    tokens = attentions[0].size(-1)
    batch_size = attentions[0].size(0)
    result = torch.eye(tokens, device=attentions[0].device).unsqueeze(0).repeat(batch_size, 1, 1)
    _debug(f"rollout attentions layers={len(attentions)} tokens={tokens}")

    for attn in attentions:
        _debug(f"layer attention shape={tuple(attn.shape)}")
        # Average heads
        attn_heads_fused = attn.mean(dim=1)
        # Optionally drop low attention values
        if discard_ratio > 0:
            flat = attn_heads_fused.view(attn_heads_fused.size(0), -1)
            _, indices = torch.sort(flat, descending=True)
            num_discard = int(flat.size(1) * discard_ratio)
            if num_discard > 0:
                for i in range(attn_heads_fused.size(0)):
                    flat[i, indices[i, -num_discard:]] = 0
                attn_heads_fused = flat.view_as(attn_heads_fused)

        # Add identity and normalize
        attn_heads_fused = attn_heads_fused + torch.eye(tokens, device=attn_heads_fused.device)
        attn_heads_fused = attn_heads_fused / attn_heads_fused.sum(dim=-1, keepdim=True)
        result = torch.bmm(attn_heads_fused, result)

    mask = result[:, 0, 1:]
    _debug(f"rollout mask shape={tuple(mask.shape)} min={mask.min().item():.6f} max={mask.max().item():.6f}")
    return mask


def rollout_to_heatmap(mask, grid_size: int):
    heatmap = mask.reshape(grid_size, grid_size).detach().cpu().numpy()
    heatmap = _normalize_map(heatmap)
    return heatmap


def attention_map_from_last_layer(attentions):
    if attentions is None or len(attentions) == 0:
        return None
    last = attentions[-1]  # [B, heads, tokens, tokens]
    _debug(f"last layer shape={tuple(last.shape)}")
    # Average across heads
    last = last.mean(dim=1)
    # CLS token attention to patch tokens
    cls_to_patches = last[:, 0, 1:]
    grid_size = int(cls_to_patches.size(-1) ** 0.5)
    heatmap = rollout_to_heatmap(cls_to_patches[0], grid_size)
    _debug(f"attention map grid={grid_size} min={heatmap.min():.6f} max={heatmap.max():.6f}")
    return heatmap


def resize_attention_map(attn_map: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    # output_size: (width, height)
    resized = cv2.resize(attn_map.astype(np.float32), output_size, interpolation=cv2.INTER_CUBIC)
    resized = _normalize_map(resized)
    return resized


def generate_attention_map(image, model, processor=None, device: str = "cpu"):
    # Helper API requested for direct debugging usage.
    if processor is None:
        from transformers import ViTImageProcessor

        processor = ViTImageProcessor.from_pretrained(model.config._name_or_path)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, output_attentions=True, return_dict=True)

    attn_map = attention_map_from_last_layer(outputs.attentions)
    if attn_map is None:
        return None

    resized = resize_attention_map(attn_map, image.size)
    from explainability.gradcam import overlay_heatmap

    return overlay_heatmap(np.array(image), resized)


def generate_attention_rollout(image, model, processor=None, device: str = "cpu"):
    # Helper API requested for direct debugging usage.
    if processor is None:
        from transformers import ViTImageProcessor

        processor = ViTImageProcessor.from_pretrained(model.config._name_or_path)
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, output_attentions=True, return_dict=True)

    rollout_mask = attention_rollout(outputs.attentions)
    if rollout_mask is None:
        return None
    grid_size = int((rollout_mask.size(-1)) ** 0.5)
    rollout_map = rollout_to_heatmap(rollout_mask[0], grid_size)
    resized = resize_attention_map(rollout_map, image.size)
    from explainability.gradcam import overlay_heatmap

    return overlay_heatmap(np.array(image), resized)
