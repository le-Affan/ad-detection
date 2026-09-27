#!/bin/bash
# Multi-visit static experiment: 2x2 arms x 3 seeds, sequential, VAL ONLY.
#   bl = baseline scans only (v9's train set)    mv = + consistent follow-ups
#   none / inv = DEMO MSE class weighting off / inverse-frequency
# Config is v9 lr_3e-5's (resnet18w, batch 32, lr 3e-5, wd 1e-2, 60 epochs,
# patience 15, default augmentation). Rerunning resumes each run from its
# last.pt and skips runs that already finished.
#
# SEEDS and ARMS override the grid, e.g. the 8-seed R0/R2 follow-up:
#   SEEDS="45 46 47 48 49" ARMS="bl:none bl:inverse" analysis/mv/run_queue.sh
cd "$(dirname "$0")/../.."
PY=/home/ankon/projects/LYP/.venv-mri/bin/python
SEEDS=${SEEDS:-"42 43 44"}
ARMS=${ARMS:-"bl:none mv:none bl:inverse mv:inverse"}
for seed in $SEEDS; do
  for arm in $ARMS; do
    cfg=${arm/:/ }
    set -- $cfg
    name="${1}_${2:0:3}_s${seed}"
    log="analysis/mv/${name}.log"
    if grep -q "^best val loss" "$log" 2>/dev/null; then echo "skip $name"; continue; fi
    extra=()
    [ "$1" = mv ] && extra+=(--train-index dataset/mri_prep_mv/train_scans.csv)
    echo "=== $name  $(date '+%F %T')"
    $PY -u -W ignore src/mri/m3_train.py "${extra[@]}" --class-weight "$2" \
        --arch resnet18w --batch-size 32 --lr 3e-5 --epochs 60 --patience 15 \
        --ckpt-every 0 --num-workers 4 --seed "$seed" \
        --out-dir "analysis/mv/${name}" --resume >> "$log" 2>&1
    tail -1 "$log"
  done
done
echo "=== all done $(date '+%F %T')"
