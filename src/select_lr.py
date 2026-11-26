"""Pick the learning rate per (model, transform) from the LR-tuning runs using VALIDATION accuracy only.

Usage: python src/select_lr.py --models mlp_b cnn_a --transform none --lrs 3e-4 1e-3 3e-3
Writes/updates configs/selected_lr.json  {"<model>_<transform>": lr}
"""
import argparse

from common import ROOT, RUNS_DIR, SEED, load_json, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--transform", default="none")
    p.add_argument("--lrs", nargs="+", type=float, required=True)
    a = p.parse_args()

    path = ROOT / "configs" / "selected_lr.json"
    sel = load_json(path) if path.exists() else {}
    report = {}
    for m in a.models:
        scores = {}
        for lr in a.lrs:
            s = RUNS_DIR / f"{m}_{a.transform}_lr{lr:g}_s{SEED}" / "summary.json"
            if s.exists():
                scores[lr] = load_json(s)["best_val_acc"]
        if not scores:
            print(f"no tuning runs found for {m}")
            continue
        # highest val acc; ties broken towards the smaller learning rate
        best = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        sel[f"{m}_{a.transform}"] = best
        report[m] = scores
        print(f"{m}: " + ", ".join(f"lr={k:g}: val={v:.4f}" for k, v in scores.items()) + f"  -> {best:g}")
    save_json(sel, path)
    save_json(report, ROOT / "results" / f"lr_tuning_{a.transform}.json")


if __name__ == "__main__":
    main()
