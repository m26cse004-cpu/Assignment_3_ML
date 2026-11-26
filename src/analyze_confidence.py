"""Q2(c) error and confidence analysis on saved TEST predictions (from eval_test.py).

Usage:
  python src/analyze_confidence.py --preds results/test/A.npz results/test/B.npz --names RGB Gray \
         [--reference results/test/A.npz] --out results/q2c_confidence.json
For every prediction file: accuracy, per-class accuracy, mean max-softmax on correct / incorrect
predictions, ECE (15 equal-width bins), fraction of errors with confidence > 0.9, confusion
matrix. With --reference, also the per-class accuracy change and the confusion-matrix difference.
Also draws a reliability diagram (figures/q2c_reliability.png).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from common import CLASSES, FIG_DIR, save_json


def ece_and_bins(conf, correct, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            bins.append((float((lo + hi) / 2), None, 0))
            continue
        acc_b, conf_b = correct[m].mean(), conf[m].mean()
        ece += m.mean() * abs(acc_b - conf_b)
        bins.append((float(conf_b), float(acc_b), int(m.sum())))
    return float(ece), bins


def analyse(path):
    d = np.load(path)
    probs, pred, y = d["probs"], d["pred"], d["y"]
    conf = probs.max(1)
    correct = pred == y
    ece, bins = ece_and_bins(conf, correct)
    cm = confusion_matrix(y, pred, labels=range(10))
    per_class = (np.diag(cm) / cm.sum(1)).tolist()
    return dict(file=str(path), acc=float(correct.mean()), per_class_acc=dict(zip(CLASSES, per_class)),
                mean_conf_correct=float(conf[correct].mean()) if correct.any() else None,
                mean_conf_incorrect=float(conf[~correct].mean()) if (~correct).any() else None,
                ece_15=ece, frac_errors_conf_gt_0_9=float((conf[~correct] > 0.9).mean()) if (~correct).any() else None,
                confusion=cm.tolist(), reliability_bins=bins)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds", nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--reference", default=None)
    p.add_argument("--out", default="results/q2c_confidence.json")
    p.add_argument("--fig", default=str(FIG_DIR / "q2c_reliability.png"))
    a = p.parse_args()

    res = {n: analyse(f) for n, f in zip(a.names, a.preds)}
    if a.reference:
        ref = analyse(a.reference)
        ref_cm = np.array(ref["confusion"])
        for n in a.names:
            r = res[n]
            r["delta_per_class_acc_vs_ref"] = {c: r["per_class_acc"][c] - ref["per_class_acc"][c] for c in CLASSES}
            diff = np.array(r["confusion"]) - ref_cm
            np.fill_diagonal(diff, 0)
            top = np.dstack(np.unravel_index(np.argsort(-diff.ravel())[:5], diff.shape))[0]
            r["largest_new_confusions_vs_ref"] = [dict(true=CLASSES[i], pred=CLASSES[j], increase=int(diff[i, j]))
                                                  for i, j in top]
    save_json(res, a.out)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    for n in a.names:
        b = [(c, acc) for c, acc, k in res[n]["reliability_bins"] if acc is not None]
        ax.plot([c for c, _ in b], [acc for _, acc in b], marker="o", ms=3, label=f"{n} (ECE {res[n]['ece_15']:.3f})")
    ax.set_xlabel("confidence (max softmax)")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(a.fig, dpi=200)
    for n in a.names:
        r = res[n]
        print(f"{n}: acc={r['acc']:.4f} ECE={r['ece_15']:.4f} conf(correct)={r['mean_conf_correct']:.3f} "
              f"conf(wrong)={r['mean_conf_incorrect']:.3f} errors>0.9={r['frac_errors_conf_gt_0_9']:.3f}")


if __name__ == "__main__":
    main()
