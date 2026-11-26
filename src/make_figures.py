"""Regenerate the training-curve figures from logs only (no training, no test set).
Q2(c), Q3(b), Q4 figures are written by their own analysis scripts.

Usage: python src/make_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import FIG_DIR, ROOT, RUNS_DIR, SEED, load_json

plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})


def sel_lr():
    p = ROOT / "configs" / "selected_lr.json"
    return load_json(p) if p.exists() else {}


def run_dir(model, transform="none", seed=SEED, lr=None, suffix=""):
    lr = lr if lr is not None else sel_lr().get(f"{model}_{transform}")
    if lr is None:
        return None
    d = RUNS_DIR / f"{model}_{transform}_lr{lr:g}_s{seed}{suffix}"
    return d if (d / "log.csv").exists() else None


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200)
    plt.close(fig)
    print("wrote", FIG_DIR / name)


def fig_q1a_curves():
    models = [("mlp_b", "MLP-B"), ("mlp_b_plain", "MLP-B-plain"), ("cnn_a", "CNN-A")]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.4), sharey=True)
    for m, label in models:
        d = run_dir(m)
        if d is None:
            continue
        df = pd.read_csv(d / "log.csv")
        for ax, col in zip(axes, ["train_acc", "val_acc"]):
            ax.plot(df.epoch, df[col], label=label)
    axes[0].set_title("training accuracy (eval mode, full 45k)")
    axes[1].set_title("validation accuracy (5k)")
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[1].legend()
    save(fig, "q1a_curves.png")


def fig_q1a_width():
    pts = []
    for m in ["mlp_s", "mlp_b", "mlp_l", "mlp_xl", "cnn_a"]:
        d = run_dir(m)
        if d is None:
            continue
        s, df = load_json(d / "summary.json"), pd.read_csv(d / "log.csv")
        pts.append((m, s["params"], s["best_val_acc"], df.train_acc.max()))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    mlp = [p for p in pts if p[0] != "cnn_a"]
    ax.plot([p[1] for p in mlp], [p[2] for p in mlp], "o-", label="MLP best val acc")
    ax.plot([p[1] for p in mlp], [p[3] for p in mlp], "s--", label="MLP max train acc", alpha=0.6)
    for p in pts:
        if p[0] == "cnn_a":
            ax.plot(p[1], p[2], "r*", ms=9, label="CNN-A best val acc")
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=6)
    save(fig, "q1a_width_sweep.png")


def fig_q1b():
    runs = sorted(RUNS_DIR.glob("mlp_mem_none_*_noise0.4"))
    if not runs:
        return
    df = pd.read_csv(runs[0] / "log.csv")
    t_star = int(df.loc[df.val_acc.idxmax(), "epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))
    ax = axes[0]
    ax.plot(df.epoch, df.clean_fit_acc, label="train, clean-label subset")
    ax.plot(df.epoch, df.noisy_fit_acc, label="train, noisy subset vs ASSIGNED label (memorisation)")
    ax.plot(df.epoch, df.noisy_true_acc, label="train, noisy subset vs TRUE label")
    ax.plot(df.epoch, df.val_acc, label="validation (clean labels)")
    ax.axvline(t_star, color="k", ls=":", lw=0.8)
    ax.text(t_star, 0.02, f" t*={t_star}", fontsize=6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("40% label noise")
    ax.legend(fontsize=5)
    ax = axes[1]
    for pattern, label, col in [("mlp_mem_none_*_s*", "0% noise: train acc", "train_acc"),
                                ("mlp_mem_none_*_noise1", "100% noise: train acc", "train_acc")]:
        rs = [r for r in sorted(RUNS_DIR.glob(pattern)) if ("noise" in r.name) == ("noise" in pattern)]
        if rs:
            d2 = pd.read_csv(rs[0] / "log.csv")
            ax.plot(d2.epoch, d2[col], label=label)
    ax.plot(df.epoch, df.clean_fit_acc, label="40% noise: clean subset", ls="--")
    ax.plot(df.epoch, df.noisy_fit_acc, label="40% noise: noisy subset", ls="--")
    ax.set_xlabel("epoch")
    ax.set_title("speed of fitting: real structure vs noise")
    ax.legend(fontsize=5)
    save(fig, "q1b_memorisation.png")


def fig_q3a():
    runs = [RUNS_DIR / f"ae_c{c}_s{SEED}" for c in (2, 16, 128)]
    runs = [r for r in runs if (r / "log.csv").exists()]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for r in runs:
        cfg, df = load_json(r / "config.json"), pd.read_csv(r / "log.csv")
        ax.plot(df.epoch, df.val_mse, label=f"d={cfg['d']} val")
        ax.plot(df.epoch, df.train_mse, ls="--", alpha=0.5, label=f"d={cfg['d']} train (running)")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=5)
    save(fig, "q3a_mse_curves.png")


if __name__ == "__main__":
    fig_q1a_curves()
    fig_q1a_width()
    fig_q1b()
    fig_q3a()
