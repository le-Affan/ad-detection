"""
Master runner for the ADNI clinical pipeline (steps 0-5).

    python preprocessing.py                    # full pipeline
    python preprocessing.py --from-step 3      # resume from step 3
    python preprocessing.py --skip-sweep       # step5 reuses selected params

Orchestration only — no step's internal logic is modified or reimplemented
here. Steps 0-4 are pure CLI scripts and are invoked as subprocesses.
Step 5 is imported instead, because its `run()` already accepts a `grid=`
argument: that is how --skip-sweep pins the hyperparameters without
needing a new flag inside step5 itself.

Each step's output is validated against known-good shapes before the next
step runs, so a bad intermediate stops the pipeline rather than silently
propagating.
"""
import argparse
import importlib
import json
import os
import subprocess
import sys
import time

import pandas as pd

SRC_DIR = "src"
PROCESSED_DIR = "dataset/processed"
MODELS_DIR = "models"

# Selected by the step5 sweep (highest val accuracy); reused by --skip-sweep.
SELECTED_PARAMS = {"depth": [6], "l2_leaf_reg": [10], "learning_rate": [0.1]}

# Known-good shapes. A step whose output does not match stops the pipeline.
EXPECTED = {
    0: {"path": f"{PROCESSED_DIR}/step0_joined.csv", "shape": (5166, 404)},
    1: {"path": f"{PROCESSED_DIR}/step1_labeled.csv", "shape": (814, 405)},
    2: {"path": f"{PROCESSED_DIR}/step2_features.csv", "shape": (814, 37)},
}
EXPECTED_SPLIT_ROWS = {"train": 569, "val": 122, "test": 123}
STEP4_COLS = 38
STEP5_ARTIFACTS = [
    "step5_catboost.cbm",
    "step5_topk_features.json",
    "step5_shap_values.npz",
    "step5_shap_bar.png",
    "step5_metrics.json",
]

STEP_SCRIPTS = {
    0: "step0_join_tables.py",
    1: "step1_data_integrity.py",
    2: "step2_feature_selection.py",
    3: "step3_split.py",
    4: "step4_processing.py",
}


class StepFailure(RuntimeError):
    """Raised when a step errors or produces output that fails validation."""


def run_script(step):
    script = os.path.join(SRC_DIR, STEP_SCRIPTS[step])
    print(f"\n{'=' * 70}\nSTEP {step}: {script}\n{'=' * 70}", flush=True)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        raise StepFailure(f"step{step} exited with code {result.returncode}")


def check_shape(step):
    exp = EXPECTED[step]
    path = exp["path"]
    if not os.path.exists(path):
        raise StepFailure(f"step{step} produced no output at {path}")
    shape = pd.read_csv(path, dtype=str, low_memory=False).shape
    if shape != exp["shape"]:
        raise StepFailure(
            f"step{step} shape {shape} != expected {exp['shape']} ({path})"
        )
    print(f"  [ok] step{step}: {shape[0]} rows x {shape[1]} cols")
    return shape


def check_step3():
    counts = {}
    for name, expected in EXPECTED_SPLIT_ROWS.items():
        path = f"{PROCESSED_DIR}/{name}_rids.csv"
        if not os.path.exists(path):
            raise StepFailure(f"step3 produced no output at {path}")
        n = len(pd.read_csv(path))
        if n != expected:
            raise StepFailure(f"step3 {name} has {n} RIDs, expected {expected}")
        counts[name] = n
    print(f"  [ok] step3: train={counts['train']} val={counts['val']} "
          f"test={counts['test']}")
    return counts


def check_step4():
    counts = {}
    for name, expected in EXPECTED_SPLIT_ROWS.items():
        path = f"{PROCESSED_DIR}/step4_{name}.csv"
        if not os.path.exists(path):
            raise StepFailure(f"step4 produced no output at {path}")
        shape = pd.read_csv(path).shape
        if shape[0] != expected:
            raise StepFailure(
                f"step4 {name} has {shape[0]} rows, expected {expected}"
            )
        if shape[1] != STEP4_COLS:
            raise StepFailure(
                f"step4 {name} has {shape[1]} cols, expected {STEP4_COLS}"
            )
        counts[name] = shape
    print(f"  [ok] step4: train={counts['train']} val={counts['val']} "
          f"test={counts['test']}")
    return counts


def run_step5(skip_sweep, top_k):
    print(f"\n{'=' * 70}\nSTEP 5: {SRC_DIR}/step5_catboost_shap.py\n{'=' * 70}",
          flush=True)
    sys.path.insert(0, SRC_DIR)
    step5 = importlib.import_module("step5_catboost_shap")

    grid = SELECTED_PARAMS if skip_sweep else step5.PARAM_GRID
    if skip_sweep:
        pinned = {k: v[0] for k, v in SELECTED_PARAMS.items()}
        print(f"--skip-sweep: reusing selected params {pinned} "
              f"(no grid search)", flush=True)
    else:
        n = 1
        for values in grid.values():
            n *= len(values)
        print(f"Running full sweep: {n} configurations", flush=True)

    _, metrics, topk, _, _ = step5.run(
        f"{PROCESSED_DIR}/step4_train.csv",
        f"{PROCESSED_DIR}/step4_val.csv",
        f"{PROCESSED_DIR}/step4_test.csv",
        MODELS_DIR,
        top_k,
        grid=grid,
    )
    return metrics, topk


def check_step5():
    missing = [
        a for a in STEP5_ARTIFACTS if not os.path.exists(os.path.join(MODELS_DIR, a))
    ]
    if missing:
        raise StepFailure(f"step5 artifacts missing: {missing}")
    print(f"  [ok] step5: {len(STEP5_ARTIFACTS)} artifacts present in {MODELS_DIR}/")


def load_prior_state():
    """Collect stage shapes for steps that were skipped via --from-step."""
    state = {"shapes": {}, "splits": None, "step4": None, "metrics": None,
             "topk": None}
    for step in (0, 1, 2):
        path = EXPECTED[step]["path"]
        if os.path.exists(path):
            state["shapes"][step] = pd.read_csv(
                path, dtype=str, low_memory=False,
            ).shape
    if all(os.path.exists(f"{PROCESSED_DIR}/{n}_rids.csv") for n in EXPECTED_SPLIT_ROWS):
        state["splits"] = {
            n: len(pd.read_csv(f"{PROCESSED_DIR}/{n}_rids.csv"))
            for n in EXPECTED_SPLIT_ROWS
        }
    if all(os.path.exists(f"{PROCESSED_DIR}/step4_{n}.csv") for n in EXPECTED_SPLIT_ROWS):
        state["step4"] = {
            n: pd.read_csv(f"{PROCESSED_DIR}/step4_{n}.csv").shape
            for n in EXPECTED_SPLIT_ROWS
        }
    metrics_path = os.path.join(MODELS_DIR, "step5_metrics.json")
    topk_path = os.path.join(MODELS_DIR, "step5_topk_features.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as fh:
            state["metrics"] = json.load(fh)
    if os.path.exists(topk_path):
        with open(topk_path) as fh:
            state["topk"] = [
                (f["feature"], f["mean_abs_shap"]) for f in json.load(fh)["features"]
            ]
    return state


def print_summary(state, elapsed, from_step):
    print(f"\n{'=' * 70}\nPIPELINE SUMMARY\n{'=' * 70}")
    if from_step > 0:
        print(f"(resumed from step {from_step}; earlier stages read from disk)\n")

    print("Stage shapes:")
    for step in (0, 1, 2):
        shape = state["shapes"].get(step)
        name = os.path.basename(EXPECTED[step]["path"])
        if shape:
            print(f"  step{step}  {name:24s} {shape[0]:>6,} rows x {shape[1]:>3} cols")
    if state["splits"]:
        s = state["splits"]
        print(f"  step3  split                    "
              f"train={s['train']} val={s['val']} test={s['test']}")
    if state["step4"]:
        s = state["step4"]
        print(f"  step4  processed                "
              f"train={s['train'][0]} val={s['val'][0]} test={s['test'][0]} "
              f"({s['train'][1]} cols)")

    metrics = state["metrics"]
    if metrics:
        print("\nFinal accuracy:")
        print(f"  train {metrics['train_accuracy']:.4f}   "
              f"val {metrics['val_accuracy']:.4f}   "
              f"test {metrics['test_accuracy']:.4f}   "
              f"(train-test gap {metrics['train_accuracy'] - metrics['test_accuracy']:+.4f})")
        if "selected_params" in metrics:
            print(f"  params: {metrics['selected_params']} "
                  f"({metrics.get('selected_trees', '?')} trees)")

    if state["topk"]:
        print("\nTop-5 SHAP features:")
        for f, v in state["topk"][:5]:
            print(f"  {f:24s} {v:.4f}")

    mins, secs = divmod(elapsed, 60)
    print(f"\nTotal runtime: {int(mins)}m {secs:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Master runner for the ADNI clinical pipeline (steps 0-5)",
    )
    parser.add_argument("--from-step", type=int, default=0, choices=range(6),
                        help="resume from this step (earlier outputs reused)")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="step5 reuses the already-selected params instead "
                             "of re-running the grid sweep")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    start = time.time()
    state = load_prior_state()

    try:
        for step in range(args.from_step, 5):
            run_script(step)
            print(f"\nValidating step{step} output:")
            if step in EXPECTED:
                state["shapes"][step] = check_shape(step)
            elif step == 3:
                state["splits"] = check_step3()
            elif step == 4:
                state["step4"] = check_step4()

        if args.from_step <= 5:
            metrics, topk = run_step5(args.skip_sweep, args.top_k)
            print("\nValidating step5 output:")
            check_step5()
            state["metrics"] = metrics
            state["topk"] = topk

    except StepFailure as exc:
        print(f"\n{'=' * 70}\nPIPELINE FAILED\n{'=' * 70}")
        print(f"  {exc}")
        print("  Stopped; later steps were not run.")
        return 1

    print_summary(state, time.time() - start, args.from_step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
