# eval_tools/paper_composites.py
"""
Create publication-style composite figures from the plots saved by run_all_evals.py.

It supports two modes:
  1) PANEL:  One page per model with Confusion + ROC (OVR) + PR (OVR) + Reliability.
  2) GRID :  A grid of confusion matrices for multiple models on a single page.

Inputs are the PNGs already produced in runs/figures and the summary CSV in runs/tables.

Usage examples
--------------
# 1) a single composite panel for ALL9 + resnet50
python -m eval_tools.paper_composites --mode panel --phase ALL9 --model resnet50

# 2) composite panels for several models (loop at shell)
for m in resnet50 densenet121 mobilenetv3 efficientnet vgg16 vgg19; do
  python -m eval_tools.paper_composites --mode panel --phase ALL9 --model $m
done

# 3) a grid of confusion matrices comparing models on ALL9
python -m eval_tools.paper_composites --mode grid --phase ALL9 \
  --models resnet50 densenet121 mobilenetv3 efficientnet vgg16 vgg19

# 4) a bar chart comparing accuracy / macroF1 across models (from comparison_all_models.csv)
python -m eval_tools.paper_composites --mode bars --phase ALL9
"""

import os, argparse, textwrap
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
from PIL import Image

FIG_DIR = os.path.join("runs", "figures")
TAB_DIR = os.path.join("runs", "tables")

def _require_file(path: str, what: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {what}: {path}")
    return path

def _load_img(path: str):
    return Image.open(path).convert("RGBA")

def _title(ax, s: str, fontsize=12):
    ax.set_title(s, fontsize=fontsize, pad=6)

def composite_panel(phase: str, model: str, out_path: str | None = None):
    """
    Assemble Confusion + ROC + PR + Reliability into one A4-like canvas.
    """
    conf_png = _require_file(os.path.join(FIG_DIR, f"confusion_{phase}_{model}.png"), "confusion png")
    roc_png  = _require_file(os.path.join(FIG_DIR, f"roc_ovr_{phase}_{model}.png"), "roc png")
    pr_png   = _require_file(os.path.join(FIG_DIR, f"pr_ovr_{phase}_{model}.png"), "pr png")
    rel_png  = _require_file(os.path.join(FIG_DIR, f"reliability_{phase}_{model}.png"), "reliability png")

    # A4 aspect roughly 8.27 x 11.69 inches → use larger dpi for print quality
    fig = plt.figure(figsize=(8.27, 11.69), dpi=220)
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.22, wspace=0.18)

    # Top row: Confusion (full width)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.imshow(_load_img(conf_png))
    ax0.axis("off")
    _title(ax0, f"{phase} — {model}: Confusion Matrix", fontsize=14)

    # Second row: ROC (left) | PR (right)
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(_load_img(roc_png)); ax1.axis("off")
    _title(ax1, "OVR ROC", fontsize=12)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(_load_img(pr_png)); ax2.axis("off")
    _title(ax2, "OVR Precision-Recall", fontsize=12)

    # Third row: Reliability (left) | Legend / meta (right)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.imshow(_load_img(rel_png)); ax3.axis("off")
    _title(ax3, "Reliability / Calibration", fontsize=12)

    # meta panel (pull metrics from comparison csv if exists)
    ax4 = fig.add_subplot(gs[2, 1]); ax4.axis("off")
    txt = f"Phase: {phase}\nModel: {model}\n"
    comp_csv = os.path.join(TAB_DIR, "comparison_all_models.csv")
    if os.path.exists(comp_csv):
        df = pd.read_csv(comp_csv)
        hit = df[(df["phase"]==phase) & (df["model"]==model)]
        if len(hit):
            acc  = hit.iloc[0]["acc"]
            mf1  = hit.iloc[0]["macroF1"]
            txt += f"\nAccuracy: {acc:.4f}\nMacro F1: {mf1:.4f}\n"
    txt += "\nNotes:\n- Plots are automatically generated from evaluation outputs.\n" \
           "- See runs/tables/pred_*.csv for per-sample predictions."
    ax4.text(0.0, 1.0, textwrap.fill(txt, 42), va="top", fontsize=10)

    if out_path is None:
        out_path = os.path.join(FIG_DIR, f"paper_panel_{phase}_{model}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PANEL] saved {out_path}")

def grid_confusions(phase: str, models: list[str], cols: int = 3, out_path: str | None = None):
    """
    Create a grid of confusion matrices for several models on one page.
    """
    k = len(models)
    rows = (k + cols - 1) // cols
    fig = plt.figure(figsize=(cols*5.2, rows*4.3), dpi=180)
    gs = gridspec.GridSpec(rows, cols, wspace=0.15, hspace=0.25)

    for i, m in enumerate(models):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r, c])
        conf_png = os.path.join(FIG_DIR, f"confusion_{phase}_{m}.png")
        if not os.path.exists(conf_png):
            ax.axis("off"); ax.text(0.5,0.5,f"{m}\n(no figure)", ha="center", va="center"); continue
        ax.imshow(_load_img(conf_png)); ax.axis("off"); _title(ax, m)

    # fill empty cells (if any)
    for i in range(k, rows*cols):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r, c]); ax.axis("off")

    if out_path is None:
        out_path = os.path.join(FIG_DIR, f"paper_grid_confusions_{phase}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.suptitle(f"Confusion Matrix Comparison — {phase}", y=0.995, fontsize=14)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[GRID] saved {out_path}")

def bar_compare(phase: str, out_path: str | None = None, sort_by: str = "acc"):
    """
    Bar chart comparing models on accuracy and macroF1 for a given phase.
    Requires runs/tables/comparison_all_models.csv generated by run_all_evals.py
    """
    comp_csv = _require_file(os.path.join(TAB_DIR, "comparison_all_models.csv"),
                             "comparison_all_models.csv")
    df = pd.read_csv(comp_csv)
    df = df[df["phase"] == phase].copy()
    if df.empty:
        raise ValueError(f"No rows for phase={phase} in {comp_csv}")
    df.sort_values(by=sort_by, ascending=False, inplace=True)

    fig = plt.figure(figsize=(9,5), dpi=180)
    ax1 = fig.add_subplot(111)
    x = range(len(df))
    ax1.bar(x, df["acc"], alpha=0.8, label="Accuracy")
    ax1.bar(x, df["macroF1"], alpha=0.8, label="Macro F1", bottom=0)  # separate colors auto
    ax1.set_xticks(list(x)); ax1.set_xticklabels(df["model"], rotation=45, ha="right")
    ax1.set_ylim(0, 1.0); ax1.set_ylabel("Score")
    ax1.set_title(f"Model Comparison — {phase}")
    ax1.legend()
    plt.tight_layout()
    if out_path is None:
        out_path = os.path.join(FIG_DIR, f"paper_bars_{phase}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[BARS] saved {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["panel","grid","bars"], required=True)
    ap.add_argument("--phase", required=True, help="ALL9, L1, L2, or custom")
    ap.add_argument("--model", help="single model name for --mode panel")
    ap.add_argument("--models", nargs="+", help="list for --mode grid")
    ap.add_argument("--cols", type=int, default=3, help="grid columns")
    ap.add_argument("--out", default=None, help="optional output path")
    args = ap.parse_args()

    if args.mode == "panel":
        if not args.model: raise SystemExit("--model is required for mode=panel")
        composite_panel(args.phase, args.model, args.out)

    elif args.mode == "grid":
        if not args.models: raise SystemExit("--models is required for mode=grid")
        grid_confusions(args.phase, args.models, cols=args.cols, out_path=args.out)

    elif args.mode == "bars":
        bar_compare(args.phase, out_path=args.out)

if __name__ == "__main__":
    main()

"""
python -m eval_tools.paper_composites --mode panel --phase ALL9 --model resnet50
python -m eval_tools.paper_composites --mode panel --phase ALL9 --model efficientnet
python -m eval_tools.paper_composites --mode panel --phase ALL9 --model vgg16
# ...repeat for any model/phase you want
python -m eval_tools.paper_composites --mode grid --phase ALL9 \
  --models resnet50 densenet121 mobilenetv3 efficientnet vgg16 vgg19 --cols 3
python -m eval_tools.paper_composites --mode bars --phase ALL9



"""