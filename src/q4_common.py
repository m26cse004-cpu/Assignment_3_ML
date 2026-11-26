"""Q4 shared setup: the ONE fixed representation and the ONE fixed stratified training subset.

Representation: bottleneck z of the AE with c=16 (d=256) from Q3(a), features/ae_c16_s26011.npz,
standardised with mean/std fitted on the Q4 training subset. Chosen before any Q4 model selection.
Subset: 1,000 images per class (10,000) drawn from the 45k training split with seed 26011,
saved to splits/q4_train_subset.npy and used for every SVM and clustering fit.
"""
import numpy as np

from common import FEAT_DIR, FAKE, SEED, SPLIT_DIR, stratified_subset

FEATURE_FILE = FEAT_DIR / "ae_c16_s26011.npz"
SUBSET_SIZE = 1000 if FAKE else 10000


def load_q4():
    d = np.load(FEATURE_FILE)
    ytr = d["y_train"]
    path = SPLIT_DIR / ("q4_train_subset" + ("_FAKE" if FAKE else "") + ".npy")
    if path.exists():
        sub = np.load(path)
    else:
        sub = stratified_subset(ytr, SUBSET_SIZE, seed=SEED)
        np.save(path, sub)
    Xtr = d["train"][sub]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    z = lambda a: ((a - mu) / sd).astype(np.float64)
    return dict(Xtr=z(Xtr), ytr=ytr[sub], sub_idx=sub, Xva=z(d["val"]), yva=d["y_val"],
                Xte=z(d["test"]), yte=d["y_test"])


def gamma_scale(X):
    """Numeric value of sklearn's gamma='scale' on the subset; stored and reused everywhere so the
    pairwise SVMs in Q4(b) use exactly the same kernel as the multiclass SVM."""
    return float(1.0 / (X.shape[1] * X.var()))
