# models/efficientnet_model.py
# ------------------------------------------------------------
# EfficientNet-B0 backbone wrapped for SkinBench classification.
# - Works for ALL9 (9-way), L2 (8-way), L1 (2-way), or SUBCLASS (6-way).
# - Keeps a small MLP classifier head (Linear -> ReLU -> Dropout -> Linear).
# - Pretrained on ImageNet by default.
# - Compatible with Grad-CAM by hooking: model.backbone.features[-1]
#     or model.backbone.features (last block) depending on torchvision ver.
# ------------------------------------------------------------

from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models


class EfficientNetMedical(nn.Module):
    """
    EfficientNet-B0 with a light classifier head.
    """
    def __init__(self,
                 num_classes: int = 9,
                 pretrained: bool = True,
                 drop: float = 0.30):
        super().__init__()

        # Load EfficientNet-B0 backbone
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)

        # Replace the stock classifier with Identity and add our own
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)        # (N, in_features)
        logits = self.classifier(feats) # (N, num_classes)
        return logits
