"""Train one classifier (MLP or CNN) on CIFAR10 under a given input transform.

Used by Q1(a), Q1(b), Q2(a), Q2(b), Q2(c). Writes to runs/<tag>/:
  config.json   all arguments + normalisation statistics + param/MAC counts
  log.csv       one row per epoch
  best.pt       checkpoint with the best validation accuracy (selection rule fixed in advance)
  last.pt       final-epoch checkpoint
  summary.json  best epoch, best val acc, final metrics
The test set is NOT touched here (see eval_test.py).
"""
import argparse
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (RUNS_DIR, SEED, Batches, apply_transform, channel_stats, get_data,
                    get_device, make_noisy_labels, normalize, predict_logits, save_json,
                    set_seed, stratified_subset, SPLIT_DIR)
from models import build_model, count_macs, count_params


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--transform", default="none", choices=["none", "permute", "gray", "blockshuffle"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--optim", default="adamw", choices=["adamw", "adam", "sgd"])
    p.add_argument("--sched", default="cosine", choices=["cosine", "const"])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--noise", type=float, default=0.0, help="fraction of training labels corrupted (Q1b)")
    p.add_argument("--train-frac", type=float, default=1.0, help="stratified fraction of the 45k train set")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def default_tag(a):
    t = f"{a.model}_{a.transform}_lr{a.lr:g}_s{a.seed}"
    if a.noise > 0:
        t += f"_noise{a.noise:g}"
    if a.train_frac < 1:
        t += f"_frac{a.train_frac:g}"
    return t


def main():
    a = parse()
    tag = a.tag or default_tag(a)
    out = RUNS_DIR / tag
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    dev = get_device()

    data = get_data()
    xtr, ytr = data["train"]
    xva, yva = data["val"]

    if a.train_frac < 1:
        keep = torch.from_numpy(stratified_subset(ytr.numpy(), a.train_frac))
        xtr, ytr = xtr[keep], ytr[keep]

    y_true_tr = ytr.clone()
    noisy_mask = torch.zeros(len(ytr), dtype=torch.bool)
    if a.noise > 0:
        y_noisy, mask = make_noisy_labels(ytr.numpy(), a.noise, seed=SEED)
        np.savez(SPLIT_DIR / f"noisy_labels_{a.noise:g}.npz", y_noisy=y_noisy, mask=mask, seed=SEED)
        ytr, noisy_mask = torch.from_numpy(y_noisy).long(), torch.from_numpy(mask)

    # transform -> statistics on (transformed) training images only -> normalise
    xtr, xva = apply_transform(xtr, a.transform), apply_transform(xva, a.transform)
    mean, std = channel_stats(xtr)
    xtr, xva = normalize(xtr, mean, std).to(dev), normalize(xva, mean, std).to(dev)
    ytr, yva = ytr.to(dev), yva.to(dev)

    model = build_model(a.model).to(dev)
    n_params, macs = count_params(model), count_macs(model)
    cfg = dict(vars(a), tag=tag, mean=mean.tolist(), std=std.tolist(), params=n_params, macs=macs,
               n_train=len(xtr), device=str(dev), torch=torch.__version__)
    save_json(cfg, out / "config.json")
    print(f"[{tag}] params={n_params:,} MACs={macs:,} n_train={len(xtr)} device={dev}")

    if a.optim == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    elif a.optim == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=a.wd, nesterov=True)
    loader = Batches(xtr, ytr, a.bs, shuffle=True, seed=a.seed)
    total = a.epochs * len(loader)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total) if a.sched == "cosine"
             else torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0))

    noisy_mask_d, y_true_d = noisy_mask.to(dev), y_true_tr.to(dev)
    clean_mask_d = ~noisy_mask_d
    fields = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "time_s"]
    if a.noise > 0:
        fields += ["clean_fit_acc", "noisy_fit_acc", "noisy_true_acc", "clean_loss", "noisy_loss"]
    f = open(out / "log.csv", "w", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()

    best_acc, best_ep, t0 = -1.0, -1, time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        for xb, yb in loader:
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

        # evaluation in eval mode on the FULL training set (with the labels it was trained on) and on val
        lt = predict_logits(model, xtr).to(dev)
        lv = predict_logits(model, xva).to(dev)
        per_ex = F.cross_entropy(lt, ytr, reduction="none")
        pred_t = lt.argmax(1)
        row = dict(epoch=ep, lr=opt.param_groups[0]["lr"],
                   train_loss=per_ex.mean().item(), train_acc=(pred_t == ytr).float().mean().item(),
                   val_loss=F.cross_entropy(lv, yva).item(), val_acc=(lv.argmax(1) == yva).float().mean().item(),
                   time_s=round(time.time() - t0, 1))
        if a.noise > 0:
            nan = float("nan")
            row.update(
                clean_fit_acc=(pred_t[clean_mask_d] == ytr[clean_mask_d]).float().mean().item() if clean_mask_d.any() else nan,
                noisy_fit_acc=(pred_t[noisy_mask_d] == ytr[noisy_mask_d]).float().mean().item(),
                noisy_true_acc=(pred_t[noisy_mask_d] == y_true_d[noisy_mask_d]).float().mean().item(),
                clean_loss=per_ex[clean_mask_d].mean().item() if clean_mask_d.any() else nan,
                noisy_loss=per_ex[noisy_mask_d].mean().item())
        w.writerow(row)
        f.flush()
        if row["val_acc"] > best_acc:
            best_acc, best_ep = row["val_acc"], ep
            torch.save(model.state_dict(), out / "best.pt")
        print(f"[{tag}] ep {ep:3d} train_acc {row['train_acc']:.4f} val_acc {row['val_acc']:.4f}"
              + (f" noisy_fit {row['noisy_fit_acc']:.4f}" if a.noise > 0 else ""))
    f.close()
    torch.save(model.state_dict(), out / "last.pt")
    save_json(dict(tag=tag, best_epoch=best_ep, best_val_acc=best_acc, final=row,
                   params=n_params, macs=macs, wall_s=round(time.time() - t0, 1)), out / "summary.json")


if __name__ == "__main__":
    main()
