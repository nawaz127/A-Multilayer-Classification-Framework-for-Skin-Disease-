import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def _ensure_dir(p):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _plot_xy_curves(curves, title, xlabel, ylabel, out_png, legend_loc="lower right", diagonal=False):
    plt.figure(figsize=(10, 7))
    for name, x, y in curves:
        plt.plot(x, y, label=name)
    if diagonal:
        plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.legend(loc=legend_loc)
    _ensure_dir(out_png)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def save_ovr_roc_and_pr(y_true, probs, classes, phase, model_name, fig_dir):
    """
    Handles both multiclass (C>=3) and binary (C==2) cases.
    Returns dict with 'roc_png' and 'pr_png' file paths.
    """
    classes = list(classes)
    C = len(classes)
    out = {"roc_png": None, "pr_png": None}

    # ----------- Binary (one-vs-rest reduces to single curve) -----------
    if C == 2:
        # binarize returns Nx1 array of "positive" (last class in 'classes') indicators
        Y = label_binarize(y_true, classes=list(range(C)))  # shape Nx1
        # choose positive class probability:
        if probs.ndim == 2 and probs.shape[1] >= 2:
            p_pos = probs[:, 1]
        else:
            # model returned only a single prob/logit – squeeze to 1D
            p_pos = probs.reshape(-1)

        fpr, tpr, _ = roc_curve(Y.ravel(), p_pos)
        roc_auc = auc(fpr, tpr)
        pr_prec, pr_rec, _ = precision_recall_curve(Y.ravel(), p_pos)
        ap = average_precision_score(Y.ravel(), p_pos)

        roc_png = os.path.join(fig_dir, f"roc_binary_{phase}_{model_name}.png")
        pr_png  = os.path.join(fig_dir, f"pr_binary_{phase}_{model_name}.png")

        _plot_xy_curves(
            curves=[(f"AUC={roc_auc:.3f}", fpr, tpr)],
            title=f"ROC (Binary) — Phase {phase}, {model_name}",
            xlabel="FPR", ylabel="TPR", out_png=roc_png, diagonal=True
        )
        _plot_xy_curves(
            curves=[(f"AP={ap:.3f}", pr_rec, pr_prec)],
            title=f"Precision–Recall (Binary) — Phase {phase}, {model_name}",
            xlabel="Recall", ylabel="Precision", out_png=pr_png, legend_loc="lower left"
        )
        out["roc_png"] = roc_png
        out["pr_png"]  = pr_png
        return out

    # ----------- Multiclass OVR (C >= 3) -----------
    Y = label_binarize(y_true, classes=list(range(C)))  # shape NxC
    # sanity: ensure probs has C columns
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    if probs.shape[1] != C:
        # try to broadcast or pad/truncate safely
        if probs.shape[1] > C:
            probs = probs[:, :C]
        else:
            pad = np.zeros((probs.shape[0], C - probs.shape[1]))
            probs = np.hstack([probs, pad])

    # per-class curves
    roc_curves = []
    pr_curves  = []
    for i, cname in enumerate(classes):
        fpr, tpr, _ = roc_curve(Y[:, i], probs[:, i])
        prec, rec, _ = precision_recall_curve(Y[:, i], probs[:, i])
        auc_i = auc(fpr, tpr)
        ap_i  = average_precision_score(Y[:, i], probs[:, i])
        roc_curves.append((f"{cname} (AUC={auc_i:.3f})", fpr, tpr))
        pr_curves.append((f"{cname} (AP={ap_i:.3f})", rec, prec))

    roc_png = os.path.join(fig_dir, f"roc_ovr_{phase}_{model_name}.png")
    pr_png  = os.path.join(fig_dir, f"pr_ovr_{phase}_{model_name}.png")

    _plot_xy_curves(roc_curves, f"OVR ROC — Phase {phase}, {model_name}",
                    xlabel="FPR", ylabel="TPR", out_png=roc_png, diagonal=True)
    _plot_xy_curves(pr_curves,  f"OVR PR — Phase {phase}, {model_name}",
                    xlabel="Recall", ylabel="Precision", out_png=pr_png, legend_loc="lower left")

    out["roc_png"] = roc_png
    out["pr_png"]  = pr_png
    return out
