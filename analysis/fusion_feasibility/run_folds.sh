#!/bin/bash
# Cross-fitted ALZ: 5 folds x 3 seeds, sequential. Clinical-train subjects only.
# Config is v9 lr_3e-5's (resnet18w, batch 32, lr 3e-5, wd 1e-2, 60 epochs,
# patience 15, default augmentation, no class weighting -- the 8-seed study
# found no reliable gain from it). Each fold's data root comes from
# make_folds.py; its `val` is an inner holdout of the 4 training folds and its
# `test` is the predicted fold, which m3 never trains or selects on.
# Rerunning resumes each run from its last.pt and skips finished runs.
cd "$(dirname "$0")/../.."
PY=/home/ankon/projects/LYP/.venv-mri/bin/python
D=analysis/fusion_feasibility
SEEDS=${SEEDS:-"42 43 44"}
FOLDS=${FOLDS:-"0 1 2 3 4"}
mkdir -p "$D/runs"
for fold in $FOLDS; do
  for seed in $SEEDS; do
    name="fold${fold}_s${seed}"
    log="$D/runs/${name}.log"
    if grep -q "^best val loss" "$log" 2>/dev/null; then echo "skip $name"; continue; fi
    echo "=== $name  $(date '+%F %T')"
    $PY -u -W ignore src/mri/m3_train.py --data-root "$D/folds/fold${fold}" \
        --arch resnet18w --batch-size 32 --lr 3e-5 --epochs 60 --patience 15 \
        --ckpt-every 0 --num-workers 4 --seed "$seed" \
        --out-dir "$D/runs/${name}" --resume >> "$log" 2>&1
    tail -1 "$log"
  done
done
echo "=== all done $(date '+%F %T')"
