"""Q3(a): reconstruction metrics for the three clean autoencoders and the selection rule.

Usage: python src/q3_eval_ae.py --runs runs/ae_c2_s26011 runs/ae_c16_s26011 runs/ae_c128_s26011
Selection (validation only): lowest validation MSE (clean input -> clean target) of the best
checkpoint. Writes configs/selected_ae.json and results/q3a_ae.json. Test MSE / PSNR are computed
for all AEs because Q3(c) compares reconstruction with class information on the test set; the
Q3(a) deliverable reports the selected AE's test error.
"""
import argparse
from pathlib import Path

import torch

from common import ROOT, add_gaussian_noise, fixed_noise, get_data, get_device, load_json, save_json
from models import ConvAE
from train_ae import TEST_NOISE_SEED


@torch.no_grad()
def recon_metrics(model, x_in, x_target, bs=500):
    model.eval()
    se, psnr = 0.0, []
    for i in range(0, len(x_in), bs):
        r = model(x_in[i:i + bs])
        per_img = ((r - x_target[i:i + bs]) ** 2).flatten(1).mean(1)
        se += per_img.sum().item()
        psnr.append(10 * torch.log10(1.0 / per_img.clamp_min(1e-12)))
    return dict(mse=se / len(x_in), psnr=torch.cat(psnr).mean().item())


def load_ae(run_dir, dev):
    cfg = load_json(Path(run_dir) / "config.json")
    m = ConvAE(cfg["c"]).to(dev)
    m.load_state_dict(torch.load(Path(run_dir) / "best.pt", map_location=dev))
    return m.eval(), cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    a = p.parse_args()
    dev = get_device()
    d = get_data()
    xva, xte = d["val"][0].to(dev), d["test"][0].to(dev)
    res = {}
    for r in a.runs:
        m, cfg = load_ae(r, dev)
        entry = dict(tag=cfg["tag"], c=cfg["c"], d=cfg["d"], params=cfg["params"],
                     val=recon_metrics(m, xva, xva), test=recon_metrics(m, xte, xte))
        if cfg.get("denoise"):
            noisy = add_gaussian_noise(xte.clone(), cfg["sigma"], fixed_noise(xte.shape, TEST_NOISE_SEED).to(dev))
            entry["test_noisy_input"] = recon_metrics(m, noisy, xte)
        res[cfg["tag"]] = entry
        print(f"{cfg['tag']}: d={cfg['d']} params={cfg['params']:,} val_mse={entry['val']['mse']:.6f} "
              f"test_mse={entry['test']['mse']:.6f}")
    clean = {k: v for k, v in res.items() if not k.startswith("dae")}
    if len(clean) > 1:
        sel = min(clean, key=lambda k: clean[k]["val"]["mse"])
        save_json(dict(tag=sel, c=clean[sel]["c"], rule="lowest validation MSE"), ROOT / "configs" / "selected_ae.json")
        print(f"selected (val only): {sel}")
    out = ROOT / "results" / "q3a_ae.json"
    if out.exists():
        old = load_json(out)
        old.update(res)
        res = old
    save_json(res, out)


if __name__ == "__main__":
    main()
