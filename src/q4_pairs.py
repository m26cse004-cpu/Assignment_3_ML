"""Q4(b): the three most-confused class pairs of the selected SVM on the test set.

Usage: python src/q4_pairs.py [--n-pairs 3]
For each pair (i, j) a binary SVC is fitted on the pair's subset points with the SAME kernel, C,
numeric gamma, degree and coef0 as the selected multiclass SVM, i.e. exactly the one-vs-one
classifier inside it. Shows (a) training support vectors: worst bounded SVs (alpha = C) and
free SVs (0 < alpha < C) nearest the margin, and (b) the test images with the smallest |f(x)|.
Quantifies: share of the pair's test errors inside the margin band |f| < 1, share of confident
errors |f| >= 1, and geometry (centroid distance, within-class spread, Fisher ratio).
Outputs results/q4b_pairs.json and figures/q4b_pair_<i>_<j>.png.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

from common import CLASSES, FIG_DIR, RESULTS_DIR, load_cifar_raw, get_split, load_json, save_json
from q4_common import load_q4


def geometry(Xa, Xb):
    ma, mb = Xa.mean(0), Xb.mean(0)
    sa, sb = ((Xa - ma) ** 2).sum(1).mean(), ((Xb - mb) ** 2).sum(1).mean()
    dist2 = float(((ma - mb) ** 2).sum())
    return dict(centroid_dist=float(np.sqrt(dist2)), spread_a=float(np.sqrt(sa)), spread_b=float(np.sqrt(sb)),
                fisher_ratio=float(dist2 / (sa + sb)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-pairs", type=int, default=3)
    p.add_argument("--n-show", type=int, default=8)
    a = p.parse_args()
    d = load_q4()
    sel = load_json(RESULTS_DIR / "q4a_selected.json")
    cm = np.array(sel["confusion"])
    sym = cm + cm.T
    pairs = sorted(((sym[i, j], i, j) for i in range(10) for j in range(i + 1, 10)), reverse=True)[:a.n_pairs]

    xtr_raw, ytr_full, xte_raw, _ = load_cifar_raw()
    tr_idx, _ = get_split(ytr_full)
    img_train = xtr_raw[tr_idx][d["sub_idx"]]  # images of the Q4 subset, aligned with d["Xtr"]

    out = {}
    for conf_count, i, j in pairs:
        mtr = np.isin(d["ytr"], [i, j])
        mte = np.isin(d["yte"], [i, j])
        Xp, yp = d["Xtr"][mtr], (d["ytr"][mtr] == j).astype(int)  # +1 side = class j
        Xq, yq = d["Xte"][mte], (d["yte"][mte] == j).astype(int)
        clf = SVC(kernel=sel["kernel"], C=sel["C"], gamma=sel["gamma"], degree=sel["degree"],
                  coef0=sel["coef0"], cache_size=2000).fit(Xp, yp)
        f_tr, f_te = clf.decision_function(Xp), clf.decision_function(Xq)
        s_te = np.where(yq == 1, 1, -1)
        err = np.sign(f_te) != s_te
        alpha = np.abs(clf.dual_coef_[0])
        bounded = alpha >= sel["C"] * (1 - 1e-6)
        sv = clf.support_
        margin_tr = np.where(yp[sv] == 1, 1, -1) * f_tr[sv]  # y*f(x); <1 inside margin, <0 misclassified
        worst_bounded = sv[bounded][np.argsort(margin_tr[bounded])][: a.n_show // 2]
        free_near = sv[~bounded][np.argsort(np.abs(margin_tr[~bounded] - 1))][: a.n_show // 2]
        near_test = np.argsort(np.abs(f_te))[: a.n_show]

        key = f"{CLASSES[i]}-{CLASSES[j]}"
        out[key] = dict(
            classes=[i, j], multiclass_confusions=int(conf_count),
            binary_test_acc=float(1 - err.mean()), n_sv=int(len(sv)), n_bounded_sv=int(bounded.sum()),
            sv_fraction=float(len(sv) / len(Xp)),
            frac_test_errors_inside_margin=float((np.abs(f_te[err]) < 1).mean()) if err.any() else None,
            frac_test_errors_confident=float((np.abs(f_te[err]) >= 1).mean()) if err.any() else None,
            frac_test_points_inside_margin=float((np.abs(f_te) < 1).mean()),
            geometry_train_subset=geometry(Xp[yp == 0], Xp[yp == 1]),
            shown_train_sv=[dict(alpha=float(np.abs(clf.dual_coef_[0][list(sv).index(k)])), y=CLASSES[[i, j][yp[k]]],
                                 f=float(f_tr[k])) for k in np.r_[worst_bounded, free_near]],
            shown_test=[dict(true=CLASSES[[i, j][yq[k]]], f=float(f_te[k]),
                             pred=CLASSES[[i, j][int(f_te[k] > 0)]]) for k in near_test])
        print(key, {k: v for k, v in out[key].items() if not k.startswith("shown")})

        te_imgs = xte_raw[mte]
        n = a.n_show
        fig, axes = plt.subplots(2, n, figsize=(n * 0.95, 2.6))
        for c, k in enumerate(np.r_[worst_bounded, free_near][:n]):
            axes[0, c].imshow(img_train[mtr][k])
            tag = "B" if bounded[list(sv).index(k)] else "F"
            axes[0, c].set_title(f"{CLASSES[[i, j][yp[k]]][:5]} {tag}\nf={f_tr[k]:+.2f}", fontsize=5)
        for c, k in enumerate(near_test):
            axes[1, c].imshow(te_imgs[k])
            ok = "ok" if not err[k] else "ERR"
            axes[1, c].set_title(f"{CLASSES[[i, j][yq[k]]][:5]} {ok}\nf={f_te[k]:+.2f}", fontsize=5)
        for ax in axes.ravel():
            ax.axis("off")
        fig.suptitle(f"{key}: top = training SVs (B bounded, F free); bottom = test points nearest boundary "
                     f"(f>0 -> {CLASSES[j]})", fontsize=6)
        fig.tight_layout()
        FIG_DIR.mkdir(exist_ok=True)
        fig.savefig(FIG_DIR / f"q4b_pair_{i}_{j}.png", dpi=200)
    save_json(out, RESULTS_DIR / "q4b_pairs.json")

    # LaTeX snippet with the pair figures, included by report/main.tex
    from common import ROOT
    imgs = "\n".join(f"\\includegraphics[width=0.72\\textwidth]{{q4b_pair_{i}_{j}.png}}\\\\" for _, i, j in pairs)
    snippet = ("\\begin{figure*}[!tbp]\\centering\n" + imgs + "\n\\caption{Q4(b): for each difficult pair, top: "
               "training support vectors (B bounded $\\alpha=C$, F free $0<\\alpha<C$); bottom: test images "
               "nearest the pairwise decision boundary, with decision scores $f$.}\\label{fig:q4b}\\end{figure*}\n")
    (ROOT / "report" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "report" / "tables" / "fig_q4b.tex").write_text(snippet)


if __name__ == "__main__":
    main()
