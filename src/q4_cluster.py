"""Q4(c): clustering the same fixed representation, compared with the selected SVM.

Usage: python src/q4_cluster.py
Fit (no labels) on the Q4 training subset; choose method and k on VALIDATION with label-free criteria:
  primary rule (pre-registered): highest validation silhouette (Euclidean, standardised features);
  also reported: k-means validation inertia per point, GMM validation mean log-likelihood and train BIC.
The selected model assigns the official test representations; test labels are used only afterwards
for ARI, NMI, the class x cluster contingency table, per-class purity/entropy and the per-pair
comparison with the SVM (SVM pair confusion rate vs clustering ARI on the two-class test subset).
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import CLASSES, FIG_DIR, RESULTS_DIR, SEED, load_json, save_json
from q4_common import load_q4

K_GRID = [5, 8, 10, 12, 15, 20]


def fit(method, k, X):
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
    return GaussianMixture(n_components=k, covariance_type="diag", random_state=SEED, max_iter=300).fit(X)


def main():
    d = load_q4()
    rows, models = [], {}
    for method in ["kmeans", "gmm_diag"]:
        for k in K_GRID:
            m = fit(method, k, d["Xtr"])
            lv = m.predict(d["Xva"])
            row = dict(method=method, k=k,
                       val_silhouette=float(silhouette_score(d["Xva"], lv)) if len(set(lv)) > 1 else float("nan"),
                       n_nonempty_val_clusters=int(len(set(lv))))
            if method == "kmeans":
                row["val_inertia_per_point"] = float(-m.score(d["Xva"]) / len(d["Xva"]))
            else:
                row["val_mean_loglik"] = float(m.score(d["Xva"]))
                row["train_bic"] = float(m.bic(d["Xtr"]))
            rows.append(row)
            models[(method, k)] = m
            print(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "q4c_selection_val.csv", index=False)
    best = df.sort_values(["val_silhouette", "k"], ascending=[False, True]).iloc[0]
    m = models[(best["method"], int(best["k"]))]
    k = int(best["k"])

    # ---- test assignment (no labels), then evaluation with labels
    lt, y = m.predict(d["Xte"]), d["yte"]
    ct = pd.crosstab(pd.Series(y, name="class").map(dict(enumerate(CLASSES))), pd.Series(lt, name="cluster"))
    ct = ct.reindex(index=CLASSES, columns=range(k), fill_value=0)
    ct.to_csv(RESULTS_DIR / "q4c_contingency_test.csv")
    C = ct.values
    res = dict(method=best["method"], k=k, selection_rule="max validation silhouette",
               test_ARI=float(adjusted_rand_score(y, lt)), test_NMI=float(normalized_mutual_info_score(y, lt)))
    if k == 10:
        r, c = linear_sum_assignment(-C)
        res["test_hungarian_acc"] = float(C[r, c].sum() / C.sum())
    res["test_purity"] = float(C.max(0).sum() / C.sum())

    # per class: SVM recall vs how a class spreads over clusters
    svm = np.load(RESULTS_DIR / "q4_svm_test_preds.npz")
    svm_pred = svm["pred"]
    sel = load_json(RESULTS_DIR / "q4a_selected.json")
    per_class = {}
    for ci, cname in enumerate(CLASSES):
        row = C[ci] / C[ci].sum()
        nz = row[row > 0]
        per_class[cname] = dict(svm_recall=float((svm_pred[y == ci] == ci).mean()),
                                dominant_cluster=int(row.argmax()), share_in_dominant_cluster=float(row.max()),
                                cluster_entropy_bits=float(-(nz * np.log2(nz)).sum()))
    res["per_class"] = per_class

    # per pair: SVM confusion rate vs clustering separability on the two-class test subset
    cm = np.array(sel["confusion"])
    pairs = []
    for i in range(10):
        for j in range(i + 1, 10):
            mk = np.isin(y, [i, j])
            pairs.append(dict(pair=f"{CLASSES[i]}-{CLASSES[j]}",
                              svm_pair_confusion_rate=float((cm[i, j] + cm[j, i]) / (cm[i].sum() + cm[j].sum())),
                              cluster_pair_ARI=float(adjusted_rand_score(y[mk], lt[mk]))))
    pdf = pd.DataFrame(pairs)
    pdf["svm_rank_easiest"] = pdf.svm_pair_confusion_rate.rank()
    pdf["cluster_rank_easiest"] = pdf.cluster_pair_ARI.rank(ascending=False)
    pdf["rank_gap"] = pdf.svm_rank_easiest - pdf.cluster_rank_easiest
    pdf.to_csv(RESULTS_DIR / "q4c_pairs_test.csv", index=False)
    res["largest_disagreements"] = dict(
        svm_easy_cluster_hard=pdf.sort_values("rank_gap").head(3).to_dict("records"),
        svm_hard_cluster_easy=pdf.sort_values("rank_gap", ascending=False).head(3).to_dict("records"))
    res["spearman_svm_vs_cluster_pairs"] = float(pdf.svm_pair_confusion_rate.rank().corr(
        (-pdf.cluster_pair_ARI).rank()))
    save_json(res, RESULTS_DIR / "q4c_cluster.json")
    print({k_: v for k_, v in res.items() if k_ not in ("per_class",)})

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.scatter(pdf.svm_pair_confusion_rate, pdf.cluster_pair_ARI, s=10)
    for _, r in pd.concat([pdf.sort_values("rank_gap").head(3), pdf.sort_values("rank_gap").tail(3)]).iterrows():
        ax.annotate(r.pair, (r.svm_pair_confusion_rate, r.cluster_pair_ARI), fontsize=5)
    ax.set_xlabel("SVM pair confusion rate (test)")
    ax.set_ylabel("clustering ARI on the pair (test)")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "q4c_pairs.png", dpi=200)


if __name__ == "__main__":
    main()
