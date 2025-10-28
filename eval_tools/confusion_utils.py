# eval_tools/confusion_utils.py
import os, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def save_confusion(y_true, y_pred, classes, out_png, out_csv, title):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    np.savetxt(out_csv, cm.astype(int), fmt="%d", delimiter=",")
    fig, ax = plt.subplots(figsize=(8,6), dpi=140)
    im = ax.imshow(cm, cmap="viridis")
    ax.set_xticks(np.arange(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="w" if cm[i,j] > cm.max()/2 else "k", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)
    return cm




