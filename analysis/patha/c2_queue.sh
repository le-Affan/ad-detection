#!/bin/bash
# Path A C2: 3 configurations x 3 seeds, sequential. Rerunning resumes each
# run from its last.pt and skips runs that already finished.
cd "$(dirname "$0")/../.."
PY=/home/ankon/projects/LYP/.venv-mri/bin/python
for seed in 42 43 44; do
  for cfg in "dx_last all" "dx_last last" "dx_bl all"; do
    set -- $cfg
    name="gru_${1}_${2}_s${seed}"
    if grep -q "^done\." "analysis/patha/${name}.log" 2>/dev/null; then echo "skip $name"; continue; fi
    echo "=== $name  $(date '+%F %T')"
    $PY -u src/mri/p5_train_temporal.py --target "$1" --visits "$2" --seed "$seed" \
        --name "$name" --resume >> "analysis/patha/${name}.log" 2>&1
    tail -1 "analysis/patha/${name}.log"
  done
done
echo "=== all done $(date '+%F %T')"
