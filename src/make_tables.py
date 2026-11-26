"""Write every LaTeX table of the report from logs/results, so no number is typed by hand.

Usage: python src/make_tables.py      -> report/tables/*.tex
Missing results are written as '--' so the report always compiles.
Accuracies are printed in % with two decimals; mean +- sample std over the seeds found (n shown).
"""
import json

import numpy as np
import pandas as pd

from common import ROOT, RESULTS_DIR, RUNS_DIR, SEED, load_json
from models import build_model, count_macs, count_params

TAB = ROOT / "report" / "tables"
SEEDS = [SEED, SEED + 1, SEED + 2]
NAMES = {"mlp_s": "MLP-S", "mlp_b": "MLP-B", "mlp_l": "MLP-L", "mlp_xl": "MLP-XL",
         "mlp_b_plain": "MLP-B-plain", "mlp_mem": "MLP-Mem", "cnn_a": "CNN-A"}
ARCH = {"mlp_s": "3072-128-128-10, BN", "mlp_b": "3072-256-256-10, BN", "mlp_l": "3072-512-512-10, BN",
        "mlp_xl": "3072-1024-1024-10, BN", "mlp_b_plain": "3072-256-256-10, no BN",
        "mlp_mem": "3072-2048-2048-10, no BN", "cnn_a": "conv 64-64-M-128-128-M-192-192-M, GAP, FC"}


def jload(p):
    return load_json(p) if p.exists() else None


def sel_lr():
    return jload(ROOT / "configs" / "selected_lr.json") or {}


def pct(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.2f}"


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{100 * vals[0]:.2f} (n=1)"
    return f"{100 * np.mean(vals):.2f} $\\pm$ {100 * np.std(vals, ddof=1):.2f} (n={len(vals)})"


def test_acc(model, transform, seed, test_transform=None, ckpt="best", suffix=""):
    lr = sel_lr().get(f"{model}_{transform}") or sel_lr().get(f"{model}_none")
    if lr is None:
        return None
    tag = f"{model}_{transform}_lr{lr:g}_s{seed}{suffix}"
    r = jload(RESULTS_DIR / "test" / f"{tag}__{ckpt}__{test_transform or transform}.json")
    return r["test_acc"] if r else None


def summary(model, transform="none", seed=SEED):
    lr = sel_lr().get(f"{model}_{transform}") or sel_lr().get(f"{model}_none")
    return jload(RUNS_DIR / f"{model}_{transform}_lr{lr:g}_s{seed}" / "summary.json") if lr else None


def write(name, body, cols, caption, label, wide=False):
    """wide=True -> table* spanning both columns; tables are shrunk (never enlarged) to fit."""
    TAB.mkdir(parents=True, exist_ok=True)
    env, width = ("table*", "\\textwidth") if wide else ("table", "\\columnwidth")
    tex = (f"\\begin{{{env}}}[!tbp]\\centering\\footnotesize\\setlength{{\\tabcolsep}}{{3pt}}\n"
           f"\\begin{{adjustbox}}{{max width={width}}}\n"
           f"\\begin{{tabular}}{{{cols}}}\\toprule\n{body}\\bottomrule\\end{{tabular}}\n\\end{{adjustbox}}\n"
           f"\\caption{{{caption}}}\\label{{{label}}}\\end{{{env}}}\n")
    (TAB / f"{name}.tex").write_text(tex)
    print("wrote", TAB / f"{name}.tex")


def tab_models():
    rows = "Model & Architecture & Params & MACs/img & LR \\\\\\midrule\n"
    for m in ["mlp_s", "mlp_b", "mlp_l", "mlp_xl", "mlp_b_plain", "cnn_a", "mlp_mem"]:
        net = build_model(m)
        lr = sel_lr().get(f"{m}_none")
        rows += (f"{NAMES[m]} & {ARCH[m]} & {count_params(net):,} & {count_macs(net):,} & "
                 f"{'--' if lr is None else f'{lr:g}'} \\\\\n")
    write("tab_models", rows, "lllrr", "Compared models (introduced in Q1(a), reused in Q2). Parameter "
          "and MAC counts are printed by the code. MLP-B and CNN-A are parameter-matched (within 5\\%) "
          "but not compute-matched.", "tab:models", wide=True)


def tab_q1a():
    rows = "Model & Params & Best val acc & Max train acc & Test acc (\\%) \\\\\\midrule\n"
    for m in ["mlp_s", "mlp_b", "mlp_l", "mlp_xl", "mlp_b_plain", "cnn_a"]:
        s = summary(m)
        if s is None:
            continue
        lr = sel_lr()[f"{m}_none"]
        log = pd.read_csv(RUNS_DIR / f"{m}_none_lr{lr:g}_s{SEED}" / "log.csv")
        rows += (f"{NAMES[m]} & {s['params']:,} & {pct(s['best_val_acc'])} & {pct(log.train_acc.max())} & "
                 f"{mean_std([test_acc(m, 'none', sd) for sd in SEEDS])} \\\\\n")
    write("tab_q1a", rows, "lrrrr", "Q1(a): validation/training accuracy (seed 26011, selected LR) "
          "and test accuracy (best-val checkpoint; mean $\\pm$ std over seeds).", "tab:q1a")


def tab_q2():
    rows = ("Model & Params & Clean & Pixel-permuted & Grayscale & RGB-trained on gray test$^\\dagger$ \\\\\\midrule\n")
    for m in ["mlp_b", "cnn_a"]:
        s = summary(m)
        p = f"{s['params']:,}" if s else "--"
        cells = [mean_std([test_acc(m, t, sd) for sd in SEEDS]) for t in ["none", "permute", "gray"]]
        diag = mean_std([test_acc(m, "none", sd, test_transform="gray") for sd in SEEDS])
        rows += f"{NAMES[m]} & {p} & " + " & ".join(cells) + f" & {diag} \\\\\n"
    write("tab_q2", rows, "lrcccc", "Q2: test accuracy (\\%) under the clean, spatially disturbed (fixed "
          "pixel permutation) and colour-removed conditions; each model is trained and tested with the same "
          "transform. $^\\dagger$Shift diagnostic only (train/test transforms differ).", "tab:q2", wide=True)


def tab_q1b():
    a = jload(RESULTS_DIR / "q1b_analysis.json")
    if a is None:
        return
    s, f = a["metrics_at_t_star"], a["metrics_final"]
    tag = a["run"].rstrip("/").split("/")[-1]
    tb = jload(RESULTS_DIR / "test" / f"{tag}__best__none.json")
    tl = jload(RESULTS_DIR / "test" / f"{tag}__last__none.json")
    rows = "Epoch & Clean fit & Noisy fit (assigned) & Noisy vs true & Val & Test \\\\\\midrule\n"
    rows += (f"$t^*$={a['t_star']} & {pct(s['clean_fit_acc'])} & {pct(s['noisy_fit_acc'])} & "
             f"{pct(s['noisy_true_acc'])} & {pct(s['val_acc'])} & {pct(tb['test_acc'] if tb else None)} \\\\\n")
    rows += (f"final={a['final_epoch']} & {pct(f['clean_fit_acc'])} & {pct(f['noisy_fit_acc'])} & "
             f"{pct(f['noisy_true_acc'])} & {pct(f['val_acc'])} & {pct(tl['test_acc'] if tl else None)} \\\\\\midrule\n")
    fe = {c: {t: ("not reached" if e is None else e) for t, e in v.items()}
          for c, v in a["first_epoch_reaching"].items()}
    for thr in ["0.25", "0.5", "0.9"]:
        rows += (f"\\multicolumn{{6}}{{l}}{{first epoch reaching {float(thr):.0%}: clean subset "
                 f"{fe['clean_fit_acc'][thr]}, noisy subset {fe['noisy_fit_acc'][thr]}}} \\\\\n").replace("%", "\\%")
    write("tab_q1b", rows, "lccccc", "Q1(b): MLP-Mem with 40\\% of training labels replaced by a wrong class. "
          "Accuracies in \\%.", "tab:q1b")


def tab_q3a():
    r = jload(RESULTS_DIR / "q3a_ae.json")
    sel = jload(ROOT / "configs" / "selected_ae.json")
    if r is None:
        return
    rows = "Bottleneck $d$ & Params & Val MSE & Val PSNR & Test MSE \\\\\\midrule\n"
    for k, v in sorted(r.items(), key=lambda kv: kv[1]["d"]):
        if k.startswith("dae"):
            continue
        chosen = bool(sel) and sel["tag"] == k
        test_mse = "{:.5f}".format(v["test"]["mse"]) if chosen else "--"
        rows += (f"{v['d']}{' (selected)' if chosen else ''} & {v['params']:,} & {v['val']['mse']:.5f} & "
                 f"{v['val']['psnr']:.2f} & {test_mse} \\\\\n")
    write("tab_q3a", rows, "lrrrr", "Q3(a): three convolutional AEs differing only in bottleneck size "
          "($z\\in\\mathbb{R}^{c\\times4\\times4}$, $d=16c$). Selection by validation MSE only.", "tab:q3a")


def tab_q3b():
    r = jload(RESULTS_DIR / "q3b_test.json")
    if r is None:
        return
    rows = "Input to fixed CNN-A & Test acc (\\%) & PSNR (dB) & Agreement w/ clean pred (\\%) \\\\\\midrule\n"
    for name, v in r["conditions"].items():
        ps = "--" if v["psnr_vs_clean"] is None else f"{v['psnr_vs_clean']:.2f}"
        rows += f"{name} & {pct(v['acc'])} & {ps} & {pct(v['agreement_with_clean_pred'])} \\\\\n"
    write("tab_q3b", rows, "lrrr", f"Q3(b): Gaussian noise $\\sigma={r['sigma']:g}$ on test images; the "
          "classifier is the frozen clean CNN-A from Q2(a).", "tab:q3b")


def tab_q3c():
    p, a = jload(RESULTS_DIR / "q3c_probe.json"), jload(RESULTS_DIR / "q3a_ae.json") or {}
    if p is None:
        return
    rows = "Representation & $d$ & Test MSE & Test PSNR & Linear & kNN & PCA-32 lin. \\\\\\midrule\n"
    for k, v in p.items():
        ae = a.get(k)
        mse = f"{ae['test']['mse']:.5f}" if ae else "--"
        ps = f"{ae['test']['psnr']:.2f}" if ae else "--"
        rows += (f"{k.replace('_', '-')} & {v['dim']} & {mse} & {ps} & {pct(v['linear']['test_acc'])} & "
                 f"{pct(v['knn']['test_acc'])} & {pct(v['pca32_linear']['test_acc'])} \\\\\n")
    write("tab_q3c", rows, "lrrrrrr", "Q3(c): reconstruction quality vs class information in frozen "
          "representations (test set; probe hyper-parameters chosen on validation). Accuracies in \\%.", "tab:q3c", wide=True)


def tab_q4a():
    p = RESULTS_DIR / "q4a_sweep.csv"
    if not p.exists():
        return
    df, sel = pd.read_csv(p), jload(RESULTS_DIR / "q4a_selected.json")
    rows = "Kernel & $C$ & Train acc & Val acc & SV fraction & Converged \\\\\\midrule\n"
    for _, r in df.iterrows():
        mark = "$^\\star$" if sel and r.kernel == sel["kernel"] and r.C == sel["C"] else ""
        rows += (f"{r.kernel}{mark} & {r.C:g} & {pct(r.train_acc)} & {pct(r.val_acc)} & {r.sv_fraction:.3f} & "
                 f"{'yes' if r.converged else 'no'} \\\\\n")
    cap = "Q4(a): SVM sweep on the fixed AE-256 representation (10k stratified training subset)."
    if sel:
        cap += f" $^\\star$Selected on validation; test accuracy {pct(sel['test_acc'])}\\%."
    write("tab_q4a", rows, "lrrrrc", cap, "tab:q4a")


def tab_q4c():
    r = jload(RESULTS_DIR / "q4c_cluster.json")
    p = RESULTS_DIR / "q4c_contingency_test.csv"
    if r is None or not p.exists():
        return
    ct = pd.read_csv(p, index_col=0)
    head = "Class & " + " & ".join(f"c{c}" for c in ct.columns) + " \\\\\\midrule\n"
    body = "".join(f"{i} & " + " & ".join(str(v) for v in row) + " \\\\\n" for i, row in ct.iterrows())
    write("tab_q4c", head + body, "l" + "r" * len(ct.columns),
          f"Q4(c): test class $\\times$ cluster contingency ({r['method']}, $k={r['k']}$, chosen by validation "
          f"silhouette). Test ARI={r['test_ARI']:.4f}, NMI={r['test_NMI']:.4f}.", "tab:q4c")


if __name__ == "__main__":
    for fn in [tab_models, tab_q1a, tab_q2, tab_q1b, tab_q3a, tab_q3b, tab_q3c, tab_q4a, tab_q4c]:
        try:
            fn()
        except Exception as e:  # keep going so partial results still produce tables
            print(f"{fn.__name__}: skipped ({e})")
