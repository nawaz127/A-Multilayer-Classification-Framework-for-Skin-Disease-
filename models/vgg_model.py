# models/vgg_model.py
from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models

class VGGMedical(nn.Module):
    """
    VGG16 / VGG19 with a compact classifier head.
    """
    def __init__(self,
                 num_classes: int = 9,
                 variant: str = "vgg16",
                 pretrained: bool = True,
                 drop: float = 0.40):
        super().__init__()

        v = variant.lower()
        if v not in {"vgg16", "vgg19"}:
            raise ValueError("variant must be 'vgg16' or 'vgg19'")

        if v == "vgg16":
            weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.vgg16(weights=weights)
        else:
            weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.vgg19(weights=weights)

        # IMPORTANT: use the FIRST linear layer's input size (25088),
        # because we replace the whole classifier with Identity.
        in_features = self.backbone.classifier[0].in_features  # 25088
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)        # (N, 25088)
        logits = self.classifier(feats) # (N, num_classes)
        return logits
