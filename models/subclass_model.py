# models/subclass_model.py
# ------------------------------------------------------------
# Fine-grained subclass classifier used AFTER L2 for:
#   - Eczema:     Atopic, Seborrheic
#   - Fungal:     Tinea, Candidiasis
#   - Pox:        Chickenpox, Monkeypox
# Total = 6 subclasses.
# - Uses EfficientNet-B0 backbone by default (good accuracy/size balance).
# - If you want a different backbone, feel free to swap (e.g., DenseNet).
# - For Grad-CAM, hook model.backbone.features[-1] (similar to EfficientNetMedical).
# ------------------------------------------------------------

from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models


class SubclassClassifier(nn.Module):
    """
    Fine-grained classifier (6-way) for Eczema/Fungal/Pox subclasses.
    """
    def __init__(self,
                 num_classes: int = 6,
                 pretrained: bool = True,
                 drop: float = 0.40):
        super().__init__()

        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits
