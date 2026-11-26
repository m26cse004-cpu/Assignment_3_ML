"""Q3(c): class information in FROZEN representations vs reconstruction quality.

Usage:
  python src/q3c_probe.py --ae-runs runs/ae_c2_s26011 runs/ae_c16_s26011 runs/ae_c128_s26011 \
                          --dae-run runs/dae_c16_sigma0.2_s26011
Representations: flattened bottleneck z (c x 4 x 4) of CLEAN images for each encoder, plus two
controls: a randomly initialised (untrained) encoder of the selected architecture and raw pixels.
Probes (encoders never updated):
  * linear: multinomial logistic regression (softmax layer, Adam, L2 penalty chosen on val)
  * kNN:    Euclidean on standardised features, k chosen on val
  * the same two probes after PCA to 32 dims (fitted on train) to control for dimension
Features of every AE are saved to features/<tag>.npz (reused by Q4).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from common import FEAT_DIR, ROOT, SEED, get_data, get_device, load_json, save_json, set_seed
from models import ConvAE
from q3_eval_ae import load_ae

L2_GRID = [0.0, 1e-5, 1e-4, 1e-3]
K_GRID = [5, 10, 20, 50]


@torch.no_grad()
def encode(model, x, dev, bs=500):
    model.eval()
    return torch.cat([model.encode(x[i:i + bs].to(dev)).flatten(1).cpu() for i in range(0, len(x), bs)]).numpy()


def standardise(tr, *others):
    mu, sd = tr.mean(0), tr.std(0) + 1e-6
    return [(z - mu) / sd for z in (tr, *others)]


def linear_probe(Xtr, ytr, Xva, yva, Xte, yte, dev, epochs=30, bs=256, lr=1e-3):
    Xtr, Xva, Xte = (torch.tensor(z, dtype=torch.float32, device=dev) for z in (Xtr, Xva, Xte))
    ytr_t = torch.tensor(ytr, device=dev)
    scores, models = {}, {}
    for l2 in L2_GRID:
        set_seed(SEED)
        lin = torch.nn.Linear(Xtr.shape[1], 10).to(dev)
        opt = torch.optim.Adam(lin.parameters(), lr=lr)
        n = len(Xtr)
        g = torch.Generator().manual_seed(SEED)
        for _ in range(epochs):
            perm = torch.randperm(n, generator=g).to(dev)
            for i in range(0, n, bs):
                j = perm[i:i + bs]
                loss = F.cross_entropy(lin(Xtr[j]), ytr_t[j]) + l2 * lin.weight.pow(2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            scores[l2] = float((lin(Xva).argmax(1).cpu().numpy() == yva).mean())
        models[l2] = lin
    best = sorted(scores.items(), key=lambda kv: (-kv[1], -kv[0]))[0][0]  # tie -> stronger penalty
    with torch.no_grad():
        test_acc = float((models[best](Xte).argmax(1).cpu().numpy() == yte).mean())
    return dict(val_by_l2={str(k): v for k, v in scores.items()}, l2=best, val_acc=scores[best], test_acc=test_acc)


def knn_predict(Xtr, ytr, Xq, kmax, dev, chunk=1000):
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    ytr_t = torch.tensor(ytr, device=dev)
    out = []
    for i in range(0, len(Xq), chunk):
        q = torch.tensor(Xq[i:i + chunk], dtype=torch.float32, device=dev)
        dist = torch.cdist(q, Xtr_t)
        out.append(ytr_t[dist.topk(kmax, largest=False).indices].cpu())
    return torch.cat(out).numpy()  # (n_query, kmax) neighbour labels sorted by distance


def vote(nbr, k):
    counts = np.apply_along_axis(lambda r: np.bincount(r, minlength=10), 1, nbr[:, :k])
    return counts.argmax(1)  # ties -> lowest class index (deterministic)


def knn_probe(Xtr, ytr, Xva, yva, Xte, yte, dev):
    nv = knn_predict(Xtr, ytr, Xva, max(K_GRID), dev)
    scores = {k: float((vote(nv, k) == yva).mean()) for k in K_GRID}
    best = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    nt = knn_predict(Xtr, ytr, Xte, best, dev)
    return dict(val_by_k={str(k): v for k, v in scores.items()}, k=best, val_acc=scores[best],
                test_acc=float((vote(nt, best) == yte).mean()))


def probe_all(name, Ftr, Fva, Fte, ytr, yva, yte, dev):
    Ztr, Zva, Zte = standardise(Ftr, Fva, Fte)
    r = dict(dim=int(Ftr.shape[1]),
             linear=linear_probe(Ztr, ytr, Zva, yva, Zte, yte, dev),
             knn=knn_probe(Ztr, ytr, Zva, yva, Zte, yte, dev))
    pca = PCA(32, random_state=SEED).fit(Ztr)
    Ptr, Pva, Pte = standardise(pca.transform(Ztr), pca.transform(Zva), pca.transform(Zte))
    r["pca32_linear"] = linear_probe(Ptr, ytr, Pva, yva, Pte, yte, dev)
    r["pca32_knn"] = knn_probe(Ptr, ytr, Pva, yva, Pte, yte, dev)
    print(f"{name:24s} dim={r['dim']:5d} linear test={r['linear']['test_acc']:.4f} "
          f"kNN test={r['knn']['test_acc']:.4f} PCA32-linear test={r['pca32_linear']['test_acc']:.4f}")
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ae-runs", nargs="+", required=True)
    p.add_argument("--dae-run", default=None)
    p.add_argument("--skip-pixels", action="store_true")
    a = p.parse_args()
    dev = get_device()
    d = get_data()
    (xtr, ytr), (xva, yva), (xte, yte) = d["train"], d["val"], d["test"]
    ytr, yva, yte = ytr.numpy(), yva.numpy(), yte.numpy()
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    res = {}
    for r in a.ae_runs + ([a.dae_run] if a.dae_run else []):
        m, cfg = load_ae(r, dev)
        Ftr, Fva, Fte = (encode(m, x, dev) for x in (xtr, xva, xte))
        np.savez(FEAT_DIR / f"{cfg['tag']}.npz", train=Ftr, val=Fva, test=Fte, y_train=ytr, y_val=yva, y_test=yte)
        res[cfg["tag"]] = probe_all(cfg["tag"], Ftr, Fva, Fte, ytr, yva, yte, dev)

    sel = load_json(ROOT / "configs" / "selected_ae.json")
    set_seed(SEED)
    rnd = ConvAE(sel["c"]).to(dev)  # untrained control (BN in eval mode uses its initial running stats)
    res[f"random_init_c{sel['c']}"] = probe_all(f"random_init_c{sel['c']}", *(encode(rnd, x, dev) for x in (xtr, xva, xte)),
                                                ytr, yva, yte, dev)
    if not a.skip_pixels:
        flat = lambda x: x.flatten(1).numpy()
        res["raw_pixels"] = probe_all("raw_pixels", flat(xtr), flat(xva), flat(xte), ytr, yva, yte, dev)
    save_json(res, ROOT / "results" / "q3c_probe.json")


if __name__ == "__main__":
    main()
