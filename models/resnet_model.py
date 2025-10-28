import torch
from torch import nn
from torchvision import models

class ResNet50(nn.Module):
    def __init__(self, num_classes=9, in_chans=3, dropout_rate=0.5):
        super().__init__()
        w = models.ResNet50_Weights.IMAGENET1K_V1
        self.backbone = models.resnet50(weights=w)
        # adapt first conv if grayscale
        if in_chans==1:
            with torch.no_grad():
                oldw = self.backbone.conv1.weight      # [64,3,7,7]
                self.backbone.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
                self.backbone.conv1.weight.copy_(oldw.mean(dim=1, keepdim=True))
        self.dropout = nn.Dropout(dropout_rate)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.backbone.fc(x)
