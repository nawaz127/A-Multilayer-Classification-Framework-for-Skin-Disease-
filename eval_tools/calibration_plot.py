# eval_tools/calibration_plot.py
import os, numpy as np, matplotlib.pyplot as plt

def expected_calibration_error(probs, y_true, n_bins=15):
    probs = np.asarray(probs); y_true = np.asarray(y_true)
    conf = probs.max(1); preds = probs.argmax(1)
    correct = (preds == y_true).astype(float)
    bins = np.linspace(0.,1.,n_bins+1)
    ece, N = 0.0, len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        idx = (conf > lo) & (conf <= hi)
        if idx.sum()==0: continue
        ece += (idx.sum()/N) * abs(correct[idx].mean() - conf[idx].mean())
    return float(ece)

def save_reliability(probs, y_true, phase, model_name, out_png, n_bins=15):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    conf = probs.max(1); preds = probs.argmax(1)
    correct = (preds == y_true).astype(float)
    bins = np.linspace(0.,1.,n_bins+1)
    mids, accs = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        idx = (conf > lo) & (conf <= hi)
        if idx.sum()==0: continue
        mids.append((lo+hi)/2); accs.append(correct[idx].mean())
    ece = expected_calibration_error(probs, y_true, n_bins)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6,5), dpi=140)
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.plot(mids, accs, marker='o')
    plt.xlabel("Confidence"); plt.ylabel("Accuracy")
    plt.title(f"Reliability (Phase {phase}, {model_name})\nECE={ece:.3f}")
    plt.tight_layout(); plt.savefig(out_png, bbox_inches='tight'); plt.close(fig)
