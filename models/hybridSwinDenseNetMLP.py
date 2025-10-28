import torch
from torch import nn
from torchvision import models
import timm

class HybridSwinDenseNetMLP(nn.Module):
    """Fuse Swin-Tiny (timm) with DenseNet121 features."""
    def __init__(self, num_classes=9, dropout=0.3):
        super().__init__()
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
        swin_dim = self.swin.num_features  # 768
        w = models.DenseNet121_Weights.IMAGENET1K_V1
        dnet = models.densenet121(weights=w)
        self.dense_features = dnet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        dense_dim = 1024
        fused = swin_dim + dense_dim
        self.head = nn.Sequential(
            nn.Linear(fused, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        if x.shape[-1] != 224:
            x224 = torch.nn.functional.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
        else:
            x224 = x
        s = self.swin(x224)            # [B, 768]
        d = self.pool(self.dense_features(x)).flatten(1)  # [B, 1024]
        z = torch.cat([s,d], dim=1)
        return self.head(z)
