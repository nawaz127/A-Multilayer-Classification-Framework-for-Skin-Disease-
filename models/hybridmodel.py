import torch
from torch import nn
from torchvision import models
import timm

class HybridViTCNNMLP(nn.Module):
    """Fuse ViT-B/16 (timm) features with ResNet18 features via concatenation then MLP."""
    def __init__(self, num_classes=9, dropout_rate=0.3):
        super().__init__()
        # ViT backbone
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        vit_dim = self.vit.num_features  # 768
        # ResNet18 backbone
        w = models.ResNet18_Weights.IMAGENET1K_V1
        self.res = models.resnet18(weights=w)
        self.res.fc = nn.Identity()
        res_dim = 512
        fused = vit_dim + res_dim
        self.head = nn.Sequential(
            nn.Linear(fused, 512), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        # Ensure 224 for ViT
        if x.shape[-1] != 224:
            x_v = torch.nn.functional.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
        else:
            x_v = x
        v = self.vit(x_v)                 # [B, 768]
        r = self.res(x)                   # [B, 512]
        z = torch.cat([v,r], dim=1)
        return self.head(z)
