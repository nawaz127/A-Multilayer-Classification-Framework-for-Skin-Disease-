import torch
from torch import nn
from torchvision import models

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=9, in_chans=3, dropout_rate=0.2):
        super().__init__()
        w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        self.backbone = models.mobilenet_v3_large(weights=w)
        # adapt first conv if grayscale
        if in_chans==1:
            with torch.no_grad():
                conv = self.backbone.features[0][0]  # first conv
                oldw = conv.weight  # [16,3,3,3]
                self.backbone.features[0][0] = nn.Conv2d(1, conv.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
                self.backbone.features[0][0].weight.copy_(oldw.mean(dim=1, keepdim=True))
        # replace classifier
        in_f = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_f, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        out = self.backbone.features(x)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1,1)).view(x.size(0), -1)
        out = self.dropout(out)
        return self.backbone.classifier(out)
