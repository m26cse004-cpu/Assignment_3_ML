"""Q3(a)/(b): train a convolutional autoencoder (clean) or denoising autoencoder. No labels used.

Clean AE:      python src/train_ae.py --c 16
Denoising AE:  python src/train_ae.py --c 16 --denoise --sigma 0.2
Inputs/targets are images in [0,1]; loss = per-pixel MSE. Fresh Gaussian noise is drawn for every
training batch; validation uses one fixed, seeded noise tensor (seed SEED+1), test uses seed SEED+2.
Checkpoint rule: lowest validation MSE of the training objective.
"""
import argparse
import csv
import time

import torch
import torch.nn.functional as F

from common import (RUNS_DIR, SEED, Batches, add_gaussian_noise, fixed_noise, get_data, get_device,
                    save_json, set_seed)
from models import ConvAE, count_params

VAL_NOISE_SEED, TEST_NOISE_SEED = SEED + 1, SEED + 2


@torch.no_grad()
def recon_mse(model, x_in, x_target, bs=500):
    model.eval()
    tot = 0.0
    for i in range(0, len(x_in), bs):
        tot += F.mse_loss(model(x_in[i:i + bs]), x_target[i:i + bs], reduction="sum").item()
    return tot / x_in.numel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c", type=int, required=True, help="bottleneck channels; z is c x 4 x 4")
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--tag", default=None)
    a = p.parse_args()
    tag = a.tag or (f"dae_c{a.c}_sigma{a.sigma:g}_s{a.seed}" if a.denoise else f"ae_c{a.c}_s{a.seed}")
    out = RUNS_DIR / tag
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    dev = get_device()

    d = get_data()
    xtr, xva = d["train"][0].to(dev), d["val"][0].to(dev)
    xva_in = add_gaussian_noise(xva.clone(), a.sigma, fixed_noise(xva.shape, VAL_NOISE_SEED).to(dev)) if a.denoise else xva

    model = ConvAE(a.c).to(dev)
    n_params = count_params(model)
    save_json(dict(vars(a), tag=tag, params=n_params, d=16 * a.c, device=str(dev), torch=torch.__version__),
              out / "config.json")
    print(f"[{tag}] params={n_params:,} bottleneck d={16 * a.c}")

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    loader = Batches(xtr, torch.zeros(len(xtr), device=dev), a.bs, shuffle=True, seed=a.seed)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * len(loader))
    f = open(out / "log.csv", "w", newline="")
    w = csv.DictWriter(f, fieldnames=["epoch", "train_mse", "val_mse", "val_mse_clean_input", "time_s"])
    w.writeheader()
    best, t0 = float("inf"), time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        run, n = 0.0, 0
        for xb, _ in loader:
            inp = add_gaussian_noise(xb.clone(), a.sigma, torch.randn_like(xb)) if a.denoise else xb
            loss = F.mse_loss(model(inp), xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item() * len(xb)
            n += len(xb)
        row = dict(epoch=ep, train_mse=run / n, val_mse=recon_mse(model, xva_in, xva),
                   val_mse_clean_input=recon_mse(model, xva, xva), time_s=round(time.time() - t0, 1))
        w.writerow(row)
        f.flush()
        if row["val_mse"] < best:
            best = row["val_mse"]
            torch.save(model.state_dict(), out / "best.pt")
        print(f"[{tag}] ep {ep:3d} train_mse {row['train_mse']:.5f} val_mse {row['val_mse']:.5f}")
    f.close()
    save_json(dict(tag=tag, best_val_mse=best, final=row, params=n_params), out / "summary.json")


if __name__ == "__main__":
    main()
