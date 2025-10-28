"""
train_multilayer.py
-------------------
Unified trainer for SkinBench with multiple phases and backbones.

Features
- Uses existing split if data_dir has train/val/test subfolders.
- Falls back to stratified split when split folders are not present.
- Supports phases: ALL9, L1 (binary), L2 (8-way w/o Normal), SUBCLASS (6-way).
- Added backbones: efficientnet (B0), vgg16, vgg19, subclass (EffNet head).
- AMP training, AdamW, ReduceLROnPlateau, early-stopping on val macro-F1.
- Saves best checkpoint and confusion matrix CSV; appends leaderboard row.

Run examples
------------
# Binary gate (Normal vs Abnormal)
python train_multilayer.py --data_dir data_raw --phase L1 --model resnet50

# 8-class abnormal (no Normal)
python train_multilayer.py --data_dir data_raw --phase L2 --model densenet121

# ALL9 flat baselines
python train_multilayer.py --data_dir data_raw --phase ALL9 --model efficientnet
python train_multilayer.py --data_dir data_raw --phase ALL9 --model vgg16
python train_multilayer.py --data_dir data_raw --phase ALL9 --model vgg19

# Subclasses (Eczema/Fungal/Pox -> 6-way)
python train_multilayer.py --data_dir data_raw --phase SUBCLASS --model subclass
"""

import argparse
import os
import random
from typing import Tuple, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from utils.datasets import RoutedFolder  # <- uses canonical names + SUBCLASS routing
from torch.cuda.amp import autocast, GradScaler

# ---- Existing models ----
from models.resnet_model import ResNet50
from models.DenseNet121 import DenseNet121Medical
from models.mobilenetv3 import MobileNetV3
from models.cnn_model import SimpleCNN
from models.hybridmodel import HybridViTCNNMLP
from models.hybridSwinDenseNetMLP import HybridSwinDenseNetMLP

# ---- NEW models ----
from models.efficientnet_model import EfficientNetMedical
from models.vgg_model import VGGMedical
from models.subclass_model import SubclassClassifier


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# registry -> returns ctor taking (num_classes) -> nn.Module
MODELS = {
    'resnet50':      lambda nc: ResNet50(num_classes=nc, in_chans=3),
    'densenet121':   lambda nc: DenseNet121Medical(num_classes=nc, in_chans=3),
    'mobilenetv3':   lambda nc: MobileNetV3(num_classes=nc, in_chans=3),
    'cnn':           lambda nc: SimpleCNN(num_classes=nc, in_chans=3),
    'vit_resnet':    lambda nc: HybridViTCNNMLP(num_classes=nc),
    'swin_densenet': lambda nc: HybridSwinDenseNetMLP(num_classes=nc),
    # new
    'efficientnet':  lambda nc: EfficientNetMedical(num_classes=nc, pretrained=True),
    'vgg16':         lambda nc: VGGMedical(num_classes=nc, variant='vgg16', pretrained=True),
    'vgg19':         lambda nc: VGGMedical(num_classes=nc, variant='vgg19', pretrained=True),
    'subclass':      lambda nc: SubclassClassifier(num_classes=nc, pretrained=True),
}


def _has_explicit_split(root: str) -> bool:
    """Return True if data_dir contains train/val/test subfolders."""
    return all(os.path.isdir(os.path.join(root, p)) for p in ('train', 'val', 'test'))


def _normalize_transform(img_size: int, train: bool) -> transforms.Compose:
    """Build transforms. IMPORTANT: ToTensor() BEFORE RandomErasing()."""
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.10),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])


def stratified_split(ds: RoutedFolder,
                     seed: int = 42,
                     train: float = 0.7,
                     val: float = 0.15) -> Tuple[List[int], List[int], List[int]]:
    """Stratified split on labels *without* calling __getitem__ (fast & safe)."""
    set_seed(seed)
    labels = np.array([lab for _, lab in ds.samples], dtype=np.int64)  # ensure int
    idx = np.arange(len(labels), dtype=np.int64)
    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(labels):
        cidx = idx[labels == c]
        np.random.shuffle(cidx)
        n = len(cidx)
        t = int(train * n)
        v = int(val * n)
        train_idx += cidx[:t].tolist()
        val_idx   += cidx[t:t + v].tolist()
        test_idx  += cidx[t + v:].tolist()
    return train_idx, val_idx, test_idx


def get_loaders(data_dir: str,
                phase: str,
                img: int = 256,
                bs: int = 32,
                seed: int = 42,
                keep_paths: bool = False):
    """
    Create DataLoaders for (train/val/test) for a given phase.
    - If data_dir has train/val/test, use them directly.
    - Otherwise, do a stratified split on the whole folder.
    """
    tr_tf = _normalize_transform(img, train=True)
    te_tf = _normalize_transform(img, train=False)

    if _has_explicit_split(data_dir):
        # Use user-provided split
        tr_ds = RoutedFolder(os.path.join(data_dir, 'train'), transform=tr_tf, phase=phase, keep_paths=keep_paths)
        va_ds = RoutedFolder(os.path.join(data_dir, 'val'),   transform=te_tf, phase=phase, keep_paths=keep_paths)
        te_ds = RoutedFolder(os.path.join(data_dir, 'test'),  transform=te_tf, phase=phase, keep_paths=keep_paths)
        classes = tr_ds.classes
        # Sanity: enforce same classes across splits
        assert classes == va_ds.classes == te_ds.classes, "Class list mismatch across splits!"
    else:
        # One-folder dataset -> do stratified split
        full = RoutedFolder(data_dir, transform=tr_tf, phase=phase, keep_paths=keep_paths)
        tr_idx, va_idx, te_idx = stratified_split(full, seed=seed)
        # For val/test use non-aug transforms
        tr_ds = Subset(full, tr_idx)
        va_ds = Subset(RoutedFolder(data_dir, transform=te_tf, phase=phase, keep_paths=keep_paths), va_idx)
        te_ds = Subset(RoutedFolder(data_dir, transform=te_tf, phase=phase, keep_paths=keep_paths), te_idx)
        classes = full.classes

    # ----- class weights (guard empty classes) -----
    if isinstance(tr_ds, Subset):
        ytr = np.array([tr_ds.dataset.samples[i][1] for i in tr_ds.indices], dtype=np.int64)
    else:
        ytr = np.array([lab for _, lab in tr_ds.samples], dtype=np.int64)

    num_classes = len(classes)
    counts = np.bincount(ytr, minlength=num_classes).astype(np.int64)
    if (counts == 0).any():
        missing = [classes[i] for i, c in enumerate(counts) if c == 0]
        print(f"[WARN] Some classes have 0 training samples: {missing}. "
              f"Consider changing --seed or verify your split.")
        counts[counts == 0] = 1  # avoid inf weights

    w = counts.max() / counts
    class_w = torch.tensor(w, dtype=torch.float32)

    tl = DataLoader(tr_ds, batch_size=bs, shuffle=True,  num_workers=4, pin_memory=True)
    vl = DataLoader(va_ds, batch_size=bs, shuffle=False, num_workers=2)
    te = DataLoader(te_ds, batch_size=bs, shuffle=False, num_workers=2)
    return tl, vl, te, class_w, classes


def evaluate(model: nn.Module, loader: DataLoader, device: str):
    """Compute (acc, macroF1, classification_report str, confusion_matrix ndarray)."""
    model.eval()
    preds, gts = [], []
    with torch.inference_mode():
        for batch in loader:
            x, y = batch[:2]  # (x, y) or (x, y, path)
            x = x.to(device); y = y.to(device)
            p = model(x).softmax(1).argmax(1)
            preds += p.cpu().tolist()
            gts   += y.cpu().tolist()
    macroF1 = f1_score(gts, preds, average='macro')
    acc = (np.array(preds) == np.array(gts)).mean()
    rep = classification_report(gts, preds, digits=4, zero_division=0)
    cm  = confusion_matrix(gts, preds)
    return acc, macroF1, rep, cm


def main(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---------- data ----------
    tl, vl, te, class_w, classes = get_loaders(
        args.data_dir, args.phase, args.img_size, args.batch_size, args.seed, keep_paths=False
    )
    num_classes = len(classes)

    # ---------- model ----------
    if args.model not in MODELS:
        raise KeyError(f"Unknown model '{args.model}'. Available: {list(MODELS.keys())}")
    model = MODELS[args.model](num_classes).to(device)

    # ---------- optim / loss / amp ----------
    crit   = nn.CrossEntropyLoss(weight=class_w.to(device))
    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=3, factor=0.3)
    scaler = GradScaler(enabled=(device == 'cuda'))

    # ---------- run dir ----------
    run_dir = os.path.join(args.out, args.phase, args.model)
    os.makedirs(run_dir, exist_ok=True)

    # ---------- train ----------
    best_f1 = -1.0
    no_improve = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for batch in tl:
            x, y = batch[:2]
            x = x.to(device); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device == 'cuda')):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # validate
        acc_v, f1_v, _, _ = evaluate(model, vl, device)
        sched.step(f1_v)
        print(f"Epoch {ep:03d}: val_acc={acc_v:.4f}  val_macroF1={f1_v:.4f}")

        if f1_v > best_f1:
            best_f1 = f1_v
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
        else:
            no_improve += 1
            if no_improve >= args.early_stop:
                print(f"Early stop at epoch {ep}.")
                break

    # ---------- test ----------
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))
    acc, f1, rep, cm = evaluate(model, te, device)
    print("\n=== TEST REPORT ===")
    print(rep)

    # ---------- logging ----------
    import pandas as pd
    lb = os.path.join(args.out, 'leaderboard.csv')
    row = pd.DataFrame([{
        'phase': args.phase, 'model': args.model,
        'img': args.img_size, 'batch': args.batch_size, 'seed': args.seed,
        'val_macroF1': best_f1, 'test_acc': acc, 'test_macroF1': f1
    }])
    if os.path.exists(lb): row.to_csv(lb, mode='a', header=False, index=False)
    else: row.to_csv(lb, index=False)

    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        os.path.join(args.out, f'confusion_{args.phase}_{args.model}.csv')
    )


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help="Root with images. If it contains train/val/test, that split is used.")
    ap.add_argument('--phase', choices=['ALL9', 'L1', 'L2', 'SUBCLASS'], default='ALL9')
    ap.add_argument('--model', choices=list(MODELS.keys()), default='resnet50')
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--early_stop', type=int, default=8)
    ap.add_argument('--out', default='runs')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    main(args)
