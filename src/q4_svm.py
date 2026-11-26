"""Q4(a): C x kernel sweep on the fixed representation, selection on validation, one test evaluation.

Usage: python src/q4_svm.py
Kernels: linear, RBF, polynomial (degree 3, coef0=1). C in {0.01, 0.1, 1, 10, 100}. gamma fixed at
the numeric value of 'scale' on the subset. Records train acc (on the subset), val acc, support-vector
fraction, fit time and whether libsvm converged. Selection: highest val acc, ties -> fewer SVs -> smaller C.
Outputs results/q4a_sweep.csv, results/q4a_selected.json, results/q4_svm_test_preds.npz,
figures/q4a_sweep.png and runs/q4_svm_selected.joblib.
"""
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from common import FIG_DIR, RESULTS_DIR, RUNS_DIR, save_json
from q4_common import gamma_scale, load_q4

C_GRID = [0.01, 0.1, 1, 10, 100]
KERNELS = ["linear", "rbf", "poly"]
MAX_ITER = 2_000_000  # safety cap; any run hitting it is flagged converged=False and must be reported


def make_svc(kernel, C, gamma):
    return SVC(kernel=kernel, C=C, gamma=gamma, degree=3, coef0=1.0, cache_size=2000,
               max_iter=MAX_ITER, decision_function_shape="ovo")


def main():
    d = load_q4()
    g = gamma_scale(d["Xtr"])
    rows = []
    for k in KERNELS:
        for C in C_GRID:
            clf = make_svc(k, C, g)
            t0 = time.time()
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter("always", ConvergenceWarning)
                clf.fit(d["Xtr"], d["ytr"])
            conv = not any(issubclass(w.category, ConvergenceWarning) for w in wl)
            row = dict(kernel=k, C=C, train_acc=float((clf.predict(d["Xtr"]) == d["ytr"]).mean()),
                       val_acc=float((clf.predict(d["Xva"]) == d["yva"]).mean()),
                       sv_fraction=len(clf.support_) / len(d["Xtr"]), n_sv=int(len(clf.support_)),
                       fit_s=round(time.time() - t0, 1), converged=conv)
            rows.append(row)
            print(row)
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "q4a_sweep.csv", index=False)

    best = df.sort_values(["val_acc", "n_sv", "C"], ascending=[False, True, True]).iloc[0]
    clf = make_svc(best["kernel"], float(best["C"]), g).fit(d["Xtr"], d["ytr"])
    pred = clf.predict(d["Xte"])
    dec = clf.decision_function(d["Xte"])
    test_acc = float((pred == d["yte"]).mean())
    save_json(dict(kernel=best["kernel"], C=float(best["C"]), gamma=g, coef0=1.0, degree=3,
                   val_acc=float(best["val_acc"]), train_acc=float(best["train_acc"]),
                   sv_fraction=float(best["sv_fraction"]), test_acc=test_acc,
                   n_train_subset=len(d["Xtr"]), feature_dim=int(d["Xtr"].shape[1]),
                   confusion=confusion_matrix(d["yte"], pred, labels=range(10)).tolist()),
              RESULTS_DIR / "q4a_selected.json")
    np.savez(RESULTS_DIR / "q4_svm_test_preds.npz", pred=pred, y=d["yte"], decision_ovo=dec)
    joblib.dump(clf, RUNS_DIR / "q4_svm_selected.joblib")
    print(f"selected {best['kernel']} C={best['C']} val={best['val_acc']:.4f} -> test={test_acc:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True)
    for ax, k in zip(axes, KERNELS):
        s = df[df.kernel == k]
        ax.plot(s.C, s.val_acc, "o-", label="val acc")
        ax.plot(s.C, s.train_acc, "s--", label="train acc", alpha=0.6)
        ax.plot(s.C, s.sv_fraction, "^-", label="SV fraction")
        ax.set_xscale("log")
        ax.set_title(k, fontsize=8)
        ax.set_xlabel("C")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "q4a_sweep.png", dpi=200)


if __name__ == "__main__":
    main()
