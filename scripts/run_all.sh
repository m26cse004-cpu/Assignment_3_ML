#!/usr/bin/env bash
# Exact reproduction commands, one stage per question part.
#   bash scripts/run_all.sh <stage> [<stage> ...]     e.g.  bash scripts/run_all.sh setup q1a_tune
#   bash scripts/run_all.sh all
# Environment overrides (state them in the report if you change them):
#   EPOCHS=60   classifier epochs         AE_EPOCHS=50    autoencoder epochs
#   MEM_EPOCHS=200  Q1(b) epochs          SEEDS="26011 26012 26013"
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-60}
AE_EPOCHS=${AE_EPOCHS:-50}
MEM_EPOCHS=${MEM_EPOCHS:-200}
SEEDS=${SEEDS:-"26011 26012 26013"}
LRS="3e-4 1e-3 3e-3"
PY=python

lr_of() {  # selected learning rate for <model>_<transform>
  $PY -c "import json;print(json.load(open('configs/selected_lr.json'))['$1'])"
}
train() {  # train <model> <transform> <lr> <seed> [extra args]; skips runs that already finished
  local m=$1 t=$2 lr=$3 s=$4; shift 4
  local tag
  tag=$($PY -c "print(f'${m}_${t}_lr{float(\"$lr\"):g}_s${s}')")
  if [ -f "runs/$tag/summary.json" ] && [ $# -eq 0 ]; then echo "skip $tag"; return; fi
  $PY src/train_classifier.py --model "$m" --transform "$t" --lr "$lr" --seed "$s" --epochs "$EPOCHS" "$@"
}
run_tag() { $PY -c "print(f'runs/${1}_${2}_lr{float(\"$3\"):g}_s${4}')"; }

stage_setup() {
  $PY src/prepare_data.py
  $PY src/models.py | tee results/param_counts.txt
}

# ---------------- Q1(a) + Q2(a): equal LR-tuning budget on validation, seed 26011
stage_q1a_tune() {
  for m in mlp_s mlp_b mlp_l mlp_xl cnn_a; do
    for lr in $LRS; do train $m none $lr 26011; done
  done
  $PY src/select_lr.py --models mlp_s mlp_b mlp_l mlp_xl cnn_a --transform none --lrs $LRS
  # weak-optimisation control: no BN, untuned default LR
  train mlp_b_plain none 1e-3 26011
  $PY src/select_lr.py --models mlp_b_plain --transform none --lrs 1e-3
}

# ---------------- Q1(a)/Q2(a) final seeds (seed 26011 is the tuning run itself)
stage_q1a_final() {
  for m in mlp_b cnn_a; do
    for s in $SEEDS; do train $m none "$(lr_of ${m}_none)" $s; done
  done
}

# ---------------- Q2(b): fixed pixel permutation, same hyper-parameters as clean
stage_q2b() {
  for m in mlp_b cnn_a; do
    for s in $SEEDS; do train $m permute "$(lr_of ${m}_none)" $s; done
  done
}

# ---------------- Q2(c): grayscale (train and test), same hyper-parameters
stage_q2c() {
  for m in mlp_b cnn_a; do
    for s in $SEEDS; do train $m gray "$(lr_of ${m}_none)" $s; done
  done
}

# ---------------- Q1(b): memorisation with 40% wrong labels + 0% / 100% controls
stage_q1b() {
  for nz in 0.4 0.0 1.0; do
    $PY src/train_classifier.py --model mlp_mem --transform none --lr 3e-4 --optim adam --wd 0 \
        --sched const --epochs "$MEM_EPOCHS" --seed 26011 --noise $nz
  done
  local run=runs/mlp_mem_none_lr0.0003_s26011_noise0.4
  $PY src/q1b_analysis.py --run $run
  $PY src/eval_test.py --runs $run --ckpt best
  $PY src/eval_test.py --runs $run --ckpt last
}

# ---------------- test-set evaluation of all selected classifiers (run once, after all choices are fixed)
stage_test() {
  local runs=()
  for m in mlp_s mlp_l mlp_xl mlp_b_plain; do runs+=("$(run_tag $m none "$(lr_of ${m}_none)" 26011)"); done
  for m in mlp_b cnn_a; do
    for t in none permute gray; do
      for s in $SEEDS; do
        r=$(run_tag $m $t "$(lr_of ${m}_none)" $s); [ -d "$r" ] && runs+=("$r")
      done
    done
  done
  $PY src/eval_test.py --runs "${runs[@]}"
  # shift diagnostic for Q2(c): RGB-trained models evaluated on gray test images
  local diag=()
  for m in mlp_b cnn_a; do for s in $SEEDS; do diag+=("$(run_tag $m none "$(lr_of ${m}_none)" $s)"); done; done
  $PY src/eval_test.py --runs "${diag[@]}" --test-transform gray
  # Q2(c) confidence / error analysis (CNN-A, seed 26011)
  local base; base=$(basename "$(run_tag cnn_a none "$(lr_of cnn_a_none)" 26011)")
  local gray; gray=$(basename "$(run_tag cnn_a gray "$(lr_of cnn_a_none)" 26011)")
  $PY src/analyze_confidence.py \
      --preds results/test/${base}__best__none.npz results/test/${gray}__best__gray.npz results/test/${base}__best__gray.npz \
      --names "RGB->RGB" "Gray->Gray" "RGB->Gray (diagnostic)" \
      --reference results/test/${base}__best__none.npz --out results/q2c_confidence.json
}

# ---------------- Q3(a): three bottlenecks (d = 32, 256, 2048)
stage_q3a() {
  for c in 2 16 128; do
    [ -f runs/ae_c${c}_s26011/summary.json ] || $PY src/train_ae.py --c $c --epochs "$AE_EPOCHS"
  done
  $PY src/q3_eval_ae.py --runs runs/ae_c2_s26011 runs/ae_c16_s26011 runs/ae_c128_s26011
}

# ---------------- Q3(b): denoising AE (same architecture as the selected clean AE)
stage_q3b() {
  local cnn; cnn=$(run_tag cnn_a none "$(lr_of cnn_a_none)" 26011)
  $PY src/q3b_eval.py --mode sigma_scan --cnn-run $cnn          # validation only, justifies sigma
  local c; c=$($PY -c "import json;print(json.load(open('configs/selected_ae.json'))['c'])")
  $PY src/train_ae.py --c $c --denoise --sigma 0.2 --epochs "$AE_EPOCHS"
  $PY src/q3_eval_ae.py --runs runs/dae_c${c}_sigma0.2_s26011
  $PY src/q3b_eval.py --mode test --cnn-run $cnn --ae-run runs/ae_c${c}_s26011 --dae-run runs/dae_c${c}_sigma0.2_s26011
}

# ---------------- Q3(c): frozen-encoder probes (also saves features used by Q4)
stage_q3c() {
  local c; c=$($PY -c "import json;print(json.load(open('configs/selected_ae.json'))['c'])")
  $PY src/q3c_probe.py --ae-runs runs/ae_c2_s26011 runs/ae_c16_s26011 runs/ae_c128_s26011 \
      --dae-run runs/dae_c${c}_sigma0.2_s26011
}

# ---------------- Q4 (bonus): SVM sweep, difficult pairs, clustering
stage_q4() {
  $PY src/q4_svm.py
  $PY src/q4_pairs.py
  $PY src/q4_cluster.py
}

stage_report() {
  $PY src/make_figures.py
  $PY src/make_tables.py
}

stage_all() {
  stage_setup; stage_q1a_tune; stage_q1a_final; stage_q2b; stage_q2c; stage_q1b
  stage_q3a; stage_test; stage_q3b; stage_q3c; stage_q4; stage_report
}

mkdir -p results configs runs
for st in "$@"; do "stage_$st"; done
