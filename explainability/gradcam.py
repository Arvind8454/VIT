from __future__ import annotations

import numpy as np
import torch
import cv2


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4):
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(alpha * heatmap + (1 - alpha) * image)
    return overlay


class ViTGradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        if target_layer is None:
            target_layer = model.vit.embeddings.patch_embeddings.projection
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output

        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, pixel_values: torch.Tensor, class_idx: int | None = None):
        # Ensure input gradients exist to make backward hooks behave as expected.
        pixel_values = pixel_values.clone().detach().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
        score = logits[:, class_idx].sum()
        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.squeeze().detach().cpu().numpy()
        return cam
