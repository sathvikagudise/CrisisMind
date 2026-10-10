"""Grad-CAM for EfficientNet-B0.

Hooks the last convolutional layer (features.8[0]) to capture
activation maps and gradients, then produces a class-discriminative
localisation heatmap.
"""

import torch
import torch.nn.functional as F

TARGET_LAYER = "features.8"


class GradCAM:
    """Grad-CAM for any EfficientNet-B0-based model.

    Hooks the *last convolutional sub-module* of ``features.8``
    (the Conv2d layer inside Conv2dNormActivation) so that both
    forward and backward hooks fire reliably.

    Usage::

        cam = GradCAM(model, target_layer="features.8")
        heatmap = cam.generate(img_tensor, class_idx=None)
    """

    def __init__(self, model, target_layer=TARGET_LAYER):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _find_sub_module(self):
        """Walk ``model.target_layer`` and return the last named child.

        For ``features.8`` (a Conv2dNormActivation) this returns the
        inner ``Conv2d`` so hooks fire correctly.
        """
        parts = self.target_layer.split(".")
        mod = self.model
        for p in parts:
            if p.isdigit():
                mod = mod[int(p)]
            else:
                mod = getattr(mod, p)
        # Drill into the last sub-container to find a leaf nn.Module
        # (Conv2dNormActivation -> Conv2d, BatchNorm2d, SiLU)
        while list(mod.children()):
            mod = next(iter(mod.children()))
        return mod

    def _register_hooks(self):
        # Hook the *first child* of the target layer (the actual Conv2d)
        # so that backward hook fires (register_full_backward_hook).
        parts = self.target_layer.split(".")
        mod = self.model
        for p in parts:
            if p.isdigit():
                mod = mod[int(p)]
            else:
                mod = getattr(mod, p)

        # Hook the Conv2d sub-module directly
        conv = mod[0] if hasattr(mod, "__getitem__") else mod
        self._hook_layer = conv

        conv.register_forward_hook(self._forward_hook)
        # Use full_backward_hook for PyTorch >= 1.10 compatibility
        conv.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def _compute(self, logits, class_idx):
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        activations = self.activations[0]
        gradients = self.gradients[0]

        weights = gradients.mean(dim=(1, 2), keepdim=True)
        heatmap = torch.sum(weights * activations, dim=0)
        heatmap = F.relu(heatmap)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        return heatmap.cpu().numpy()

    def generate(self, img_tensor, class_idx=None):
        output = self.model(img_tensor)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        heatmap = self._compute(logits, class_idx)
        return heatmap, class_idx

    def generate_with_logits(self, img_tensor, class_idx=None):
        output = self.model(img_tensor)
        logits = output[0] if isinstance(output, (tuple, list)) else output
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        heatmap = self._compute(logits, class_idx)
        return heatmap, class_idx, logits.detach()
