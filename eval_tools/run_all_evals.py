import os, argparse, inspect
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, f1_score, accuracy_score

from utils.datasets import RoutedFolder
# the following helpers are from your eval_tools package
from eval_tools.models_registry import REGISTRY
from eval_tools.confusion_utils import save_confusion
from eval_tools.roc_pr_curves import save_ovr_roc_and_pr
from eval_tools.calibration_plot import save_reliability

PHASES = ["ALL9", "L1", "L2"]
MODELS = [
    "resnet50","densenet121","mobilenetv3","cnn",
    "vit_resnet","swin_densenet","efficientnet","vgg16","vgg19"
]

# ---------- Dataset constructor that adapts to your RoutedFolder signature ----------
def _make_dataset(data_dir: str, phase: str, split: str, img: int):
    norm = transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
    T = transforms.Compose([transforms.Resize((img, img)),
                            transforms.ToTensor(), norm])

    sig = inspect.signature(RoutedFolder)
    params = list(sig.parameters.keys())
    kwargs = dict(transform=T, phase=phase, keep_paths=True)

    # root parameter name handling
    root_kw = None
    for cand in ("root", "data_root", "path", "data_dir"):
        if cand in params:
            root_kw = cand
            break
    # split-like parameter name handling
    split_kw = None
    for cand in ("split", "subset", "stage", "mode", "partition", "split_name"):
        if cand in params:
            split_kw = cand
            break

    # Build call
    if root_kw is not None:
        kwargs[root_kw] = data_dir
        if split_kw is not None:
            kwargs[split_kw] = split
        ds = RoutedFolder(**kwargs)
    else:
        # first parameter is likely the root; pass positionally
        if split_kw is not None:
            kwargs[split_kw] = split
        ds = RoutedFolder(data_dir, **kwargs)

    return ds

def _loader(data_dir, phase, split="test", img=256, batch=64):
    ds = _make_dataset(data_dir, phase, split, img)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    return dl, ds.classes

@torch.no_grad()
def _eval_one(phase, model_name, ckpt, data_dir, device, out_root, img=256, batch=64):
    loader, classes = _loader(data_dir, phase, 'test', img=img, batch=batch)
    C = len(classes)
    model = REGISTRY[model_name](C, device, ckpt)

    y_true, y_pred, y_prob, paths = [], [], [], []
    for x, y, p in loader:
        x = x.to(device)
        prob = torch.softmax(model(x), dim=1).cpu().numpy()
        y_prob.append(prob)
        y_pred.extend(prob.argmax(1))
        y_true.extend(y.numpy())
        paths += p

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.vstack(y_prob)

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    print(f"[EVAL] phase={phase} model={model_name} acc={acc:.4f} macroF1={mf1:.4f}")
    print(classification_report(y_true, y_pred, digits=4))

    fig_dir = os.path.join(out_root, "figures")
    tab_dir = os.path.join(out_root, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    # Confusion
    cm_png = os.path.join(fig_dir, f"confusion_{phase}_{model_name}.png")
    cm_csv = os.path.join(tab_dir, f"confusion_{phase}_{model_name}.csv")
    save_confusion(y_true, y_pred, classes, cm_png, cm_csv, f"confusion_{phase}_{model_name}")

    # ROC + PR
    rocpr = save_ovr_roc_and_pr(y_true, y_prob, classes, phase, model_name, fig_dir)

    # Reliability
    rel_png = os.path.join(fig_dir, f"reliability_{phase}_{model_name}.png")
    save_reliability(y_prob, y_true, phase, model_name, rel_png)

    # Per-sample CSV
    df = pd.DataFrame({"path": paths, "y_true": y_true, "y_pred": y_pred})
    for i, c in enumerate(classes):
        df[f"p_{c}"] = y_prob[:, i]
    pred_csv = os.path.join(tab_dir, f"pred_{phase}_{model_name}.csv")
    df.to_csv(pred_csv, index=False)

    return {
        "phase": phase, "model": model_name, "acc": acc, "macroF1": mf1,
        "cm_png": cm_png, "roc_png": rocpr["roc_png"], "pr_png": rocpr["pr_png"],
        "rel_png": rel_png, "pred_csv": pred_csv
    }

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = "runs"
    rows = []

    for phase in args.phases:
        for m in args.models:
            ckpt = os.path.join("runs", phase, m, "best.pt")
            if not os.path.exists(ckpt):
                print(f"[SKIP] {ckpt} missing")
                continue
            r = _eval_one(phase, m, ckpt, args.data_dir, device, out_root,
                          img=args.img, batch=args.batch)
            rows.append(r)

    if rows:
        os.makedirs(os.path.join(out_root, "tables"), exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(out_root, "tables", "comparison_all_models.csv"),
                                  index=False)
        print("Saved runs/tables/comparison_all_models.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--phases", nargs="+", default=PHASES)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    main(ap.parse_args())
