import torch
from torch import nn
from torchvision import models

class DenseNet121Medical(nn.Module):
    def __init__(self, num_classes=9, in_chans=3, dropout_rate=0.5):
        super().__init__()
        w = models.DenseNet121_Weights.IMAGENET1K_V1
        self.backbone = models.densenet121(weights=w)
        if in_chans==1:
            with torch.no_grad():
                # first conv lives at features.conv0
                oldw = self.backbone.features.conv0.weight  # [64,3,7,7]
                self.backbone.features.conv0 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
                self.backbone.features.conv0.weight.copy_(oldw.mean(dim=1, keepdim=True))
        self.dropout = nn.Dropout(dropout_rate)
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)
    def forward(self, x):
        feats = self.backbone.features(x)
        out = torch.relu(feats)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1,1)).view(x.size(0), -1)
        out = self.dropout(out)
        return self.backbone.classifier(out)
