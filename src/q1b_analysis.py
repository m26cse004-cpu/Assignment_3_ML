"""Q1(b): turn the per-epoch memorisation log into the numbers used in the argument.

Usage: python src/q1b_analysis.py --run runs/mlp_mem_none_lr0.0003_s26011_noise0.4
Outputs results/q1b_analysis.json with
  t_star                  epoch of peak validation accuracy (the val-selected checkpoint)
  metrics_at_t_star       clean-fit, noisy-fit (assigned wrong labels), noisy-true, val acc at t*
  metrics_final           the same at the last epoch
  first_epoch_reaching    first epoch each subset's fit reaches 25/50/90 %
  decision                the pre-registered rule applied (see PREREGISTRATION.md)
"""
import argparse
from pathlib import Path

import pandas as pd

from common import save_json

KEYS = ["clean_fit_acc", "noisy_fit_acc", "noisy_true_acc", "val_acc", "train_acc"]


def first_epoch(df, col, thr):
    hit = df.index[df[col] >= thr]
    return int(df.loc[hit[0], "epoch"]) if len(hit) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--noisy-fit-threshold", type=float, default=0.20,
                   help="pre-registered: memorisation counts as 'not yet begun' below this noisy-fit accuracy")
    p.add_argument("--out", default="results/q1b_analysis.json")
    a = p.parse_args()

    df = pd.read_csv(Path(a.run) / "log.csv")
    i_star = int(df["val_acc"].idxmax())
    at = lambda i: {k: float(df.loc[i, k]) for k in KEYS}
    res = dict(run=a.run, t_star=int(df.loc[i_star, "epoch"]), metrics_at_t_star=at(i_star),
               final_epoch=int(df["epoch"].iloc[-1]), metrics_final=at(len(df) - 1),
               first_epoch_reaching={c: {str(t): first_epoch(df, c, t) for t in (0.25, 0.5, 0.9)}
                                     for c in ["clean_fit_acc", "noisy_fit_acc"]},
               noisy_true_acc_peak=dict(epoch=int(df.loc[df["noisy_true_acc"].idxmax(), "epoch"]),
                                        value=float(df["noisy_true_acc"].max())))
    m = res["metrics_at_t_star"]
    res["decision"] = dict(
        rule=f"structure-before-memorisation iff at t*: noisy_fit_acc < {a.noisy_fit_threshold} "
             f"and noisy_true_acc > 0.10 (chance)",
        noisy_fit_below_threshold=m["noisy_fit_acc"] < a.noisy_fit_threshold,
        noisy_true_above_chance=m["noisy_true_acc"] > 0.10,
    )
    res["decision"]["supported"] = bool(res["decision"]["noisy_fit_below_threshold"]
                                        and res["decision"]["noisy_true_above_chance"])
    save_json(res, a.out)
    print(res)


if __name__ == "__main__":
    main()
