from __future__ import annotations

import numpy as np
import torch


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    return arr


def generate_integrated_gradients_map(
    model,
    pixel_values: torch.Tensor,
    class_idx: int,
    steps: int = 24,
) -> np.ndarray:
    """
    Compute an Integrated Gradients saliency map for ViT input pixels.

    Returns a normalized 2D heatmap in [0, 1].
    """
    if steps < 8:
        steps = 8

    device = pixel_values.device
    inputs = pixel_values.clone().detach()
    baseline = torch.zeros_like(inputs, device=device)
    total_grads = torch.zeros_like(inputs, device=device)

    alphas = torch.linspace(0.0, 1.0, steps=steps, device=device)
    model.zero_grad(set_to_none=True)

    param_dtype = next(model.parameters()).dtype
    inputs_for_model = inputs.to(dtype=param_dtype)
    baseline_for_model = baseline.to(dtype=param_dtype)

    for alpha in alphas:
        scaled = baseline_for_model + alpha * (inputs_for_model - baseline_for_model)
        scaled.requires_grad_(True)
        outputs = model(pixel_values=scaled, return_dict=True)
        score = outputs.logits[:, class_idx].sum()
        grads = torch.autograd.grad(score, scaled, retain_graph=False, create_graph=False)[0].float()
        total_grads = total_grads + grads.detach()

    avg_grads = total_grads / float(steps)
    integrated = (inputs.float() - baseline.float()) * avg_grads
    saliency = integrated.abs().sum(dim=1).squeeze(0).detach().cpu().numpy()
    return _normalize(saliency)
