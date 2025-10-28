# eval_tools/models_registry.py
import torch
from typing import Callable, Dict

# your models
from models.resnet_model import ResNet50
from models.DenseNet121 import DenseNet121Medical
from models.mobilenetv3 import MobileNetV3
from models.cnn_model import SimpleCNN
from models.hybridmodel import HybridViTCNNMLP
from models.hybridSwinDenseNetMLP import HybridSwinDenseNetMLP
from models.efficientnet_model import EfficientNetMedical
from models.vgg_model import VGGMedical

def _load_ckpt(model: torch.nn.Module, ckpt: str, device: str):
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # strip possible 'module.' prefixes from DDP
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

# build(name) -> callable(num_classes, device, ckpt|None)
REGISTRY: Dict[str, Callable] = {
    "resnet50":      lambda C, dev, ckpt=None: _load_ckpt(ResNet50(num_classes=C).to(dev), ckpt, dev) if ckpt else ResNet50(num_classes=C).to(dev).eval(),
    "densenet121":   lambda C, dev, ckpt=None: _load_ckpt(DenseNet121Medical(num_classes=C).to(dev), ckpt, dev) if ckpt else DenseNet121Medical(num_classes=C).to(dev).eval(),
    "mobilenetv3":   lambda C, dev, ckpt=None: _load_ckpt(MobileNetV3(num_classes=C).to(dev), ckpt, dev) if ckpt else MobileNetV3(num_classes=C).to(dev).eval(),
    "cnn":           lambda C, dev, ckpt=None: _load_ckpt(SimpleCNN(num_classes=C).to(dev), ckpt, dev) if ckpt else SimpleCNN(num_classes=C).to(dev).eval(),
    "vit_resnet":    lambda C, dev, ckpt=None: _load_ckpt(HybridViTCNNMLP(num_classes=C).to(dev), ckpt, dev) if ckpt else HybridViTCNNMLP(num_classes=C).to(dev).eval(),
    "swin_densenet": lambda C, dev, ckpt=None: _load_ckpt(HybridSwinDenseNetMLP(num_classes=C).to(dev), ckpt, dev) if ckpt else HybridSwinDenseNetMLP(num_classes=C).to(dev).eval(),
    # NEW
    "efficientnet":  lambda C, dev, ckpt=None: _load_ckpt(EfficientNetMedical(num_classes=C).to(dev), ckpt, dev) if ckpt else EfficientNetMedical(num_classes=C).to(dev).eval(),
    "vgg16":         lambda C, dev, ckpt=None: _load_ckpt(VGGMedical(num_classes=C, variant="vgg16").to(dev), ckpt, dev) if ckpt else VGGMedical(num_classes=C, variant="vgg16").to(dev).eval(),
    "vgg19":         lambda C, dev, ckpt=None: _load_ckpt(VGGMedical(num_classes=C, variant="vgg19").to(dev), ckpt, dev) if ckpt else VGGMedical(num_classes=C, variant="vgg19").to(dev).eval(),
}
