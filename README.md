# ML Assignment 3: MLPs, CNNs, Autoencoders, SVMs and Clustering on CIFAR10

**Author:** Jiya Yadav (M26CSE004) · **Split seed:** 26004 (digits of the roll number) · **Report:** [`report/a3_report.pdf`](report/a3_report.pdf)

Every number in the report is produced by the commands below. The scripts write the numbers to `runs/`,
`results/` and `report/tables/`; no number is typed in by hand.

## 1. Environment actually used

| Item | Value |
|---|---|
| Machine | Ubuntu 24.04.1 LTS lab server |
| GPU | 1 × NVIDIA RTX PRO 6000 (Blackwell), driver 595.58.03, CUDA 13.2 |
| PyTorch | 2.15.0+cu132 (`torch.cuda.is_available() == True`) |
| Python environment | `venv` with pip; exact package versions in `requirements-lock.txt` |

Setup (no conda, no sudo needed):

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt          # or, for the exact versions used: pip install -r requirements-lock.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
```

Blackwell GPUs need a recent PyTorch build (2.7 or newer). `environment.yml` is an equivalent conda
alternative, but the results were produced with the venv above.

## 2. Reproduction (the exact commands that were run)

All commands run from the repository root with the venv active. Long stages were run in the background
with `nohup` so that they survive a closed SSH session.

```bash
# data + split: downloads CIFAR10 to data/, saves splits/split_seed26011.npz, prints parameter counts
bash scripts/run_all.sh setup

# Q1(a)/Q2(a): equal LR-tuning budget, grid {3e-4, 1e-3, 3e-3}, validation accuracy, seed 26011
nohup bash scripts/run_all.sh q1a_tune > logs/q1a_tune.log 2>&1 &

# PREREGISTRATION.md was committed here (commit 2931a26), during LR tuning and before any final or test run

# everything else, in this order
nohup bash scripts/run_all.sh q1a_final q2b q2c q1b q3a test q3b q3c q4 report > logs/main.log 2>&1 &
```

What each stage does:

| Stage | Question | Content |
|---|---|---|
| `setup` | all | download, stratified 45,000/5,000 split with seed 26011, parameter counts |
| `q1a_tune` | Q1(a), Q2(a) | MLP-S/B/L/XL and CNN-A × 3 learning rates (seed 26011); MLP-B-plain at a fixed 1e-3 |
| `q1a_final` | Q1(a), Q2(a) | MLP-B and CNN-A with seeds 26011–26013 at the selected learning rate |
| `q2b` | Q2(b) | one fixed permutation of the 1,024 pixel positions, retrain both models, same hyper-parameters |
| `q2c` | Q2(c) | grayscale train/val/test, retrain both models, same hyper-parameters |
| `q1b` | Q1(b) | MLP-Mem with 40% wrong labels + 0% and 100% controls, 200 epochs; analysis; test at t* and at the last epoch |
| `q3a` | Q3(a) | conv AEs with bottleneck d = 32, 256, 2048; selection by validation MSE |
| `test` | all | the official test set, once, for all selected classifiers; Q2(c) confidence analysis |
| `q3b` | Q3(b) | sigma scan on validation, denoising AE, frozen CNN-A on the six test conditions |
| `q3c` | Q3(c) | frozen-encoder linear and kNN probes (+ PCA-32); saves features for Q4 |
| `q4` | Q4 | SVM C × kernel sweep, difficult pairs, clustering |
| `report` | report | regenerates `figures/` and `report/tables/` from the logs |

The report is compiled from `report/main.tex`:
`cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main`.

Defaults: 60 classifier epochs, 50 AE epochs, 200 Q1(b) epochs, seeds 26011–26013. The same defaults were used
for every reported number.

## 3. Main results (test set)

Mean ± sample std over 3 seeds where n = 3 is shown; other entries are single runs.

| Experiment | Result |
|---|---|
| Q1(a)/Q2(a) clean, MLP-B (856,074 params) | 55.07 ± 0.43 % |
| Q1(a)/Q2(a) clean, CNN-A (816,202 params) | 87.54 ± 0.45 % |
| Q2(b) pixel-permuted, MLP-B / CNN-A | 54.72 ± 0.20 % / 52.05 ± 0.42 % |
| Q2(c) grayscale, MLP-B / CNN-A | 46.69 ± 0.37 % / 85.37 ± 0.30 % |
| Q1(b) 40% wrong labels, at t* = 5 / at epoch 200 | 46.31 % / 35.22 % |
| Q3(a) selected AE (d = 2048), test MSE | 0.00036 |
| Q3(b) fixed CNN-A on noisy / DAE(noisy) / clean | 17.11 % / 62.11 % / 87.17 % |
| Q4(a) selected SVM (RBF, C = 10, 10k subset) | 48.04 % |
| Q4(c) k-means (k = 5), test ARI / NMI | 0.0506 / 0.0802 |

Full tables are in `report/tables/` and the raw values in `results/`.

## 4. Protocol (what is fixed, and where)

| Item | Value | Where |
|---|---|---|
| Split | stratified 45,000 / 5,000 from the 50k training images, `random_state=26011` | `src/common.py`, `splits/` |
| Normalisation | per-channel mean/std of the (transformed) 45k training images | saved in each `runs/<tag>/config.json` |
| Augmentation | none (crops/flips are a spatial prior and would confound Q1/Q2) | `src/train_classifier.py` |
| Optimiser | AdamW, weight decay 5e-4, batch 128, cosine schedule | `src/train_classifier.py` |
| Tuning budget | same LR grid for every compared model, validation accuracy, seed 26011 | `src/select_lr.py`, `configs/selected_lr.json` |
| Checkpoint rule | epoch with the best validation accuracy (AEs: lowest validation MSE) | `best.pt` |
| Test set | used only after selection, by `eval_test.py`, `q3_eval_ae.py`, `q3b_eval.py --mode test`, `q3c_probe.py`, `q4_*.py` | `results/test/` |
| Q2(b) transform | one fixed permutation of the 1,024 pixel positions, `default_rng(26011)`, same for all channels and splits | `apply_transform` |
| Q2(c) transform | Y = 0.299R + 0.587G + 0.114B, replicated to 3 channels | `apply_transform` |
| Q1(b) noise | 40% of training labels, each replaced by a uniformly drawn *other* class, seed 26011 | `splits/noisy_labels_0.4.npz` |
| Q3(b) corruption | additive Gaussian, sigma = 0.2 on [0,1], clipped; fresh per batch in training; fixed seeds 26012 (val) / 26013 (test) | `src/train_ae.py` |
| Q4 representation | AE bottleneck z, d = 256 (`ae_c16`, Q3(a)), standardised on the subset; fixed in `PREREGISTRATION.md` | `src/q4_common.py` |
| Q4 subset | stratified 10,000 (1,000 per class) of the 45k, seed 26011, fixed before model selection | `splits/q4_train_subset.npy` |

## 5. Reuse map

| Component | Introduced in | Reused in |
|---|---|---|
| Split, normalisation, training loop (`common.py`, `train_classifier.py`) | Q1(a) | Q1(b), Q2, Q3(b) |
| MLP-B, CNN-A and their checkpoints | Q1(a) | Q2(a) (same runs); CNN-A seed 26011 is the fixed classifier in Q3(b) |
| AEs with d = 32 / 256 / 2048 | Q3(a) | Q3(b) (clean-AE baseline), Q3(c), Q4 (d = 256 features) |
| Denoising AE | Q3(b) | Q3(c) |
| Selected SVM and its test confusion matrix | Q4(a) | Q4(b), Q4(c) |

## 6. Repository layout

```
src/            all code (one script per experiment; see the docstring at the top of each file)
scripts/        run_all.sh: the stages above
splits/         saved split, noisy labels, Q4 subset
configs/        selected_lr.json, selected_ae.json (written by the selection scripts, validation only)
runs/<tag>/     config.json, log.csv (one row per epoch), summary.json   [checkpoints *.pt are not committed]
logs/           full console logs of the two runs above
results/        test results and analysis JSON/CSV
figures/        every figure, regenerated by the scripts
report/         main.tex, references.bib, tables/ (auto-generated), a3_report.pdf
PREREGISTRATION.md   fairness criteria and predictions, committed before the final runs (2931a26)
```

Checkpoints, extracted features and the dataset are not committed because of their size. The commands above
regenerate all of them.

## 7. Smoke test without CIFAR10

`A3_FAKE_DATA=1 EPOCHS=2 AE_EPOCHS=2 MEM_EPOCHS=3 SEEDS="26011" bash scripts/run_all.sh all` runs the whole
pipeline on a tiny synthetic dataset, to check the code only. Run it in a separate copy of the repository,
because it writes its own (meaningless) results to `runs/`, `results/` and `configs/`.
