# eval_tools/subclass_eval.py
import os, argparse, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, f1_score, accuracy_score
from utils.datasets import RoutedFolder
from eval_tools.models_registry import REGISTRY
from eval_tools.confusion_utils import save_confusion
from eval_tools.roc_pr_curves import save_ovr_roc_and_pr

TARGETS = ["Eczema","Fungal Infections","Pox"]

def _loader(root, top, split="test", img=256, batch=64):
    norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    T = transforms.Compose([transforms.Resize((img,img)), transforms.ToTensor(), norm])
    # RoutedFolder supports 'phase'; here we pass the top disease to restrict routing.
    ds = RoutedFolder(root, transform=T, phase=top, split=split, keep_paths=True)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0), ds.classes

@torch.no_grad()
def _eval_one(root, top, model_name, device):
    ckpt = os.path.join("runs","L3",top,model_name,"best.pt")
    if not os.path.exists(ckpt):
        print(f"[SKIP] {top}/{model_name} -> {ckpt} not found")
        return None
    loader, classes = _loader(root, top, "test")
    C = len(classes)
    model = REGISTRY[model_name](C, device, ckpt)

    y, yhat, prob, paths = [], [], [], []
    for x,t,p in loader:
        x = x.to(device)
        pr = torch.softmax(model(x), 1).cpu().numpy()
        prob.append(pr); yhat.extend(pr.argmax(1)); y.extend(t.numpy()); paths += p
    y, yhat, prob = np.array(y), np.array(yhat), np.vstack(prob)
    acc = accuracy_score(y,yhat); mf1 = f1_score(y,yhat, average="macro")
    print(f"[L3] {top} {model_name} acc={acc:.4f} macroF1={mf1:.4f}")
    outF = "runs/figures"; outT = "runs/tables"
    os.makedirs(outF, exist_ok=True); os.makedirs(outT, exist_ok=True)
    save_confusion(y,yhat,classes, os.path.join(outF,f"confusion_L3_{top}_{model_name}.png"),
                   os.path.join(outT,f"confusion_L3_{top}_{model_name}.csv"),
                   f"confusion_L3_{top}_{model_name}")
    save_ovr_roc_and_pr(y, prob, classes, f"L3-{top}", model_name, outF)
    print(classification_report(y,yhat, digits=4))
    pd.DataFrame({"path":paths,"y_true":y,"y_pred":yhat}).to_csv(os.path.join(outT,f"pred_L3_{top}_{model_name}.csv"), index=False)
    return {"top":top,"model":model_name,"acc":acc,"macroF1":mf1}

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for top in args.top:
        for m in args.models:
            r = _eval_one(args.data_sub, top, m, device)
            if r: rows.append(r)
    if rows:
        os.makedirs("runs/tables", exist_ok=True)
        pd.DataFrame(rows).to_csv("runs/tables/comparison_L3.csv", index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_sub", required=True)
    ap.add_argument("--top", nargs="+", default=TARGETS)
    ap.add_argument("--models", nargs="+", default=["resnet50","densenet121","efficientnet","vgg16","vgg19"])
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    main(ap.parse_args())
