"""The ONLY script that evaluates classifiers on the official CIFAR10 test set.

Usage:
  python src/eval_test.py --runs runs/cnn_a_none_lr0.001_s26011 [more runs] [--ckpt best|last]
         [--test-transform gray]   # optional: shift diagnostic (e.g. RGB-trained model on gray test)
By default the test images get exactly the transform used in training (and the training
normalisation statistics). Outputs results/test/<tag>__<ckpt>__<test-transform>.{json,npz}.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from common import (RESULTS_DIR, apply_transform, get_data, get_device, load_json, normalize,
                    predict_logits, save_json)
from models import build_model


def evaluate(run_dir, ckpt="best", test_transform=None):
    run_dir = Path(run_dir)
    cfg = load_json(run_dir / "config.json")
    dev = get_device()
    tt = test_transform or cfg["transform"]
    xte, yte = get_data()["test"]
    x = normalize(apply_transform(xte, tt), torch.tensor(cfg["mean"]), torch.tensor(cfg["std"])).to(dev)
    model = build_model(cfg["model"]).to(dev)
    model.load_state_dict(torch.load(run_dir / f"{ckpt}.pt", map_location=dev))
    logits = predict_logits(model, x)
    probs = torch.softmax(logits, 1).numpy()
    pred = probs.argmax(1)
    y = yte.numpy()
    acc = float((pred == y).mean())
    name = f"{cfg['tag']}__{ckpt}__{tt}"
    out = RESULTS_DIR / "test"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", probs=probs, pred=pred, y=y)
    res = dict(tag=cfg["tag"], model=cfg["model"], train_transform=cfg["transform"], test_transform=tt,
               ckpt=ckpt, seed=cfg["seed"], lr=cfg["lr"], noise=cfg.get("noise", 0.0),
               params=cfg["params"], test_acc=acc, n_test=int(len(y)))
    save_json(res, out / f"{name}.json")
    print(f"{name}: test_acc={acc:.4f}")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--ckpt", default="best", choices=["best", "last"])
    p.add_argument("--test-transform", default=None)
    a = p.parse_args()
    for r in a.runs:
        evaluate(r, a.ckpt, a.test_transform)


if __name__ == "__main__":
    main()
