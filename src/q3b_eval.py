"""Q3(b): does denoising recover the information the (fixed) classifier needs?

1) Noise-level justification on VALIDATION only:
   python src/q3b_eval.py --mode sigma_scan --cnn-run runs/cnn_a_none_lr..._s26011
2) Test-set comparison with the frozen clean CNN-A:
   python src/q3b_eval.py --mode test --cnn-run ... --ae-run runs/ae_c16_s26011 --dae-run runs/dae_c16_sigma0.2_s26011
Conditions: clean, noisy, AE(noisy), DAE(noisy), AE(clean) [blur control], DAE(clean).
Reports accuracy, per-class accuracy, PSNR vs the clean image, and agreement with the clean prediction.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (CLASSES, FIG_DIR, ROOT, add_gaussian_noise, fixed_noise, get_data, get_device,
                    load_json, normalize, predict_logits, save_json)
from models import build_model
from q3_eval_ae import load_ae
from train_ae import TEST_NOISE_SEED, VAL_NOISE_SEED


def load_cnn(run_dir, dev):
    cfg = load_json(Path(run_dir) / "config.json")
    assert cfg["transform"] == "none", "the fixed classifier must be the clean-RGB CNN"
    m = build_model(cfg["model"]).to(dev)
    m.load_state_dict(torch.load(Path(run_dir) / "best.pt", map_location=dev))
    return m.eval(), torch.tensor(cfg["mean"], device=dev), torch.tensor(cfg["std"], device=dev)


@torch.no_grad()
def reconstruct(ae, x, bs=500):
    return torch.cat([ae(x[i:i + bs]) for i in range(0, len(x), bs)])


def psnr(x, ref):
    mse = ((x - ref) ** 2).flatten(1).mean(1).clamp_min(1e-12)
    return float((10 * torch.log10(1.0 / mse)).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sigma_scan", "test"], required=True)
    p.add_argument("--cnn-run", required=True)
    p.add_argument("--ae-run")
    p.add_argument("--dae-run")
    p.add_argument("--sigmas", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3])
    a = p.parse_args()
    dev = get_device()
    cnn, mean, std = load_cnn(a.cnn_run, dev)
    d = get_data()
    classify = lambda x: predict_logits(cnn, normalize(x, mean, std)).argmax(1).numpy()

    if a.mode == "sigma_scan":
        xva, yva = d["val"][0].to(dev), d["val"][1].numpy()
        base = fixed_noise(xva.shape, VAL_NOISE_SEED).to(dev)
        res = {"clean": float((classify(xva) == yva).mean())}
        for s in a.sigmas:
            res[f"sigma={s:g}"] = float((classify(add_gaussian_noise(xva.clone(), s, base)) == yva).mean())
        save_json(res, ROOT / "results" / "q3b_sigma_scan_val.json")
        print(res)
        return

    dae, dcfg = load_ae(a.dae_run, dev)
    ae, _ = load_ae(a.ae_run, dev)
    sigma = dcfg["sigma"]
    xte, yte = d["test"][0].to(dev), d["test"][1].numpy()
    noisy = add_gaussian_noise(xte.clone(), sigma, fixed_noise(xte.shape, TEST_NOISE_SEED).to(dev))
    conds = {"clean": xte, "noisy": noisy, "AE(noisy)": reconstruct(ae, noisy), "DAE(noisy)": reconstruct(dae, noisy),
             "AE(clean)": reconstruct(ae, xte), "DAE(clean)": reconstruct(dae, xte)}
    pred_clean = classify(xte)
    res = dict(sigma=sigma, cnn_run=a.cnn_run, ae_run=a.ae_run, dae_run=a.dae_run, conditions={})
    for name, x in conds.items():
        pr = classify(x)
        res["conditions"][name] = dict(
            acc=float((pr == yte).mean()), psnr_vs_clean=psnr(x, xte) if name != "clean" else None,
            agreement_with_clean_pred=float((pr == pred_clean).mean()),
            per_class_acc={c: float((pr[yte == k] == k).mean()) for k, c in enumerate(CLASSES)})
        print(f"{name:12s} acc={res['conditions'][name]['acc']:.4f}")
    save_json(res, ROOT / "results" / "q3b_test.json")

    # qualitative figure: 8 test images x 4 rows
    idx = np.random.default_rng(0).choice(len(xte), 8, replace=False)
    rows = ["clean", "noisy", "AE(noisy)", "DAE(noisy)"]
    fig, axes = plt.subplots(len(rows), 8, figsize=(8, 4.3))
    for r, name in enumerate(rows):
        for c, i in enumerate(idx):
            axes[r, c].imshow(conds[name][i].permute(1, 2, 0).cpu().numpy())
            axes[r, c].axis("off")
        axes[r, 0].set_title(name, fontsize=7, loc="left")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "q3b_examples.png", dpi=200)


if __name__ == "__main__":
    main()
