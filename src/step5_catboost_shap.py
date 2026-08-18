"""
Step 5 — CatBoost classifier + SHAP feature importance.

Input:  dataset/processed/step4_{train,val,test}.csv (RID + label + 35
        features + CogIndex; PTGENDER categorical, rest already normalized)
Output: models/step5_catboost.cbm            trained model
        models/step5_topk_features.json      Top-K features by mean |SHAP|
        models/step5_shap_values.npz         SHAP values, per split
        models/step5_shap_bar.png            bar plot, test-set SHAP
        models/step5_metrics.json            train/val/test accuracy,
                                              test per-class precision/recall/F1

Trains CatBoost MultiClass (4-class: CN/EMCI/LMCI/AD) on train, using val
for eval/early stopping, PTGENDER passed as a categorical feature (not
one-hot encoded). SHAP TreeExplainer runs against the test set.

Hyperparameters are chosen by a small sweep over depth / l2_leaf_reg /
learning_rate, selected on VAL accuracy only — the test split is not
consulted until the winning configuration is fixed.
"""
import argparse
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ID_COLUMNS = ["RID", "label"]
CATEGORICAL_FEATURES = ["PTGENDER"]
LABEL_NAMES = {0: "CN", 1: "EMCI", 2: "LMCI", 3: "AD"}
SEED = 42
TOP_K = 20

# Sweep grid. The defaults (depth 6, l2_leaf_reg 3) overfit badly on this
# 814-subject cohort: 99.1% train vs 79.7% test. The grid biases toward
# shallower trees and stronger L2 to close that gap.
PARAM_GRID = {
    "depth": [4, 5, 6],
    "l2_leaf_reg": [3, 5, 10],
    "learning_rate": [0.03, 0.1],
}


def load_split(path):
    df = pd.read_csv(path)
    X = df.drop(columns=ID_COLUMNS)
    y = df["label"]
    return df, X, y


def train_model(train_pool, val_pool, cat_features, params):
    model = CatBoostClassifier(
        loss_function="MultiClass",
        random_seed=SEED,
        cat_features=cat_features,
        early_stopping_rounds=50,
        iterations=1000,
        verbose=False,
        **params,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model


def sweep(train_pool, val_pool, cat_features, X_train, y_train, X_val, y_val, grid):
    """Grid search selected on VAL accuracy only — test is never consulted.

    Ties on val accuracy are broken toward the more regularized model
    (shallower, then stronger L2), so a tie never silently picks the
    higher-capacity config.
    """
    names = list(grid)
    results = []
    for combo in itertools.product(*(grid[n] for n in names)):
        params = dict(zip(names, combo))
        model = train_model(train_pool, val_pool, cat_features, params)
        results.append({
            "params": params,
            "train_accuracy": float(accuracy_score(y_train, model.predict(X_train))),
            "val_accuracy": float(accuracy_score(y_val, model.predict(X_val))),
            "trees": int(model.tree_count_),
            "model": model,
        })

    best = max(
        results,
        key=lambda r: (
            r["val_accuracy"],
            -r["params"]["depth"],
            r["params"]["l2_leaf_reg"],
        ),
    )
    return best, results


def compute_metrics(model, X_train, y_train, X_val, y_val, X_test, y_test):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    test_pred = model.predict(X_test).ravel()
    test_acc = accuracy_score(y_test, test_pred)

    classes = sorted(LABEL_NAMES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, test_pred, labels=classes, zero_division=0,
    )
    per_class = {
        LABEL_NAMES[c]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, c in enumerate(classes)
    }

    return {
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "test_per_class": per_class,
    }


def compute_shap(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # For MultiClass CatBoost, shap.TreeExplainer returns an array shaped
    # (n_samples, n_features, n_classes); collapse to per-feature importance
    # by taking the mean |SHAP| across samples AND classes.
    shap_arr = np.asarray(shap_values)
    if shap_arr.ndim == 3:
        mean_abs = np.abs(shap_arr).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(shap_arr).mean(axis=0)
    return shap_arr, mean_abs


def run(train_path, val_path, test_path, out_dir, top_k, grid=PARAM_GRID):
    train_df, X_train, y_train = load_split(train_path)
    val_df, X_val, y_val = load_split(val_path)
    test_df, X_test, y_test = load_split(test_path)

    cat_features = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    best, sweep_results = sweep(
        train_pool, val_pool, cat_features, X_train, y_train, X_val, y_val, grid,
    )
    # The winning run is already fit on train with the selected params and
    # early-stopped on val, so it is reused directly — refitting with the
    # same seed and data would reproduce it exactly.
    model = best["model"]
    metrics = compute_metrics(model, X_train, y_train, X_val, y_val, X_test, y_test)
    metrics["selected_params"] = best["params"]
    metrics["selected_trees"] = best["trees"]
    metrics["selection_criterion"] = "highest val accuracy; test not consulted"
    metrics["sweep"] = [
        {k: r[k] for k in ("params", "train_accuracy", "val_accuracy", "trees")}
        for r in sweep_results
    ]

    shap_arr, mean_abs = compute_shap(model, X_test)
    ranked = sorted(
        zip(X_test.columns, mean_abs), key=lambda kv: kv[1], reverse=True,
    )
    topk = ranked[:top_k]

    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "step5_catboost.cbm")
    model.save_model(model_path)

    topk_path = os.path.join(out_dir, "step5_topk_features.json")
    with open(topk_path, "w") as fh:
        json.dump(
            {"k": top_k, "features": [{"feature": f, "mean_abs_shap": float(v)} for f, v in topk]},
            fh, indent=2,
        )

    shap_path = os.path.join(out_dir, "step5_shap_values.npz")
    np.savez(
        shap_path,
        shap_values=shap_arr,
        feature_names=np.array(X_test.columns),
        test_rids=test_df["RID"].to_numpy(),
    )

    plot_path = os.path.join(out_dir, "step5_shap_bar.png")
    top_features, top_values = zip(*topk)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(topk))))
    ax.barh(range(len(topk)), top_values[::-1])
    ax.set_yticks(range(len(topk)))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(f"Top-{top_k} features by mean |SHAP| (test set)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    metrics_path = os.path.join(out_dir, "step5_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    paths = {
        "model": model_path,
        "topk": topk_path,
        "shap_values": shap_path,
        "shap_bar": plot_path,
        "metrics": metrics_path,
    }
    return model, metrics, topk, paths, X_train.columns.tolist()


def main():
    parser = argparse.ArgumentParser(description="Step 5: CatBoost + SHAP")
    parser.add_argument("--train-path", default="dataset/processed/step4_train.csv")
    parser.add_argument("--val-path", default="dataset/processed/step4_val.csv")
    parser.add_argument("--test-path", default="dataset/processed/step4_test.csv")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    model, metrics, topk, paths, feature_cols = run(
        args.train_path, args.val_path, args.test_path, args.out_dir, args.top_k,
    )

    print("Output written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    print(f"\nSweep: {len(metrics['sweep'])} configurations, "
          f"selected on val accuracy only (test not consulted).")
    print(f"Selected params: {metrics['selected_params']} "
          f"({metrics['selected_trees']} trees after early stopping)")
    print("\nSweep results (sorted by val accuracy):")
    for r in sorted(metrics["sweep"], key=lambda r: r["val_accuracy"], reverse=True):
        p = r["params"]
        print(f"  depth={p['depth']} l2={p['l2_leaf_reg']:2d} lr={p['learning_rate']:.2f}  "
              f"train={r['train_accuracy']:.4f} val={r['val_accuracy']:.4f} "
              f"gap={r['train_accuracy'] - r['val_accuracy']:+.4f} trees={r['trees']}")

    print(f"\nFeature matrix passed to CatBoost: {len(feature_cols)} columns "
          f"(RID and label excluded -- confirmed: dropped via ID_COLUMNS "
          f"before fit).")
    print(f"Feature columns: {feature_cols}")

    print("\nAccuracy:")
    print(f"  train: {metrics['train_accuracy']:.4f}")
    print(f"  val:   {metrics['val_accuracy']:.4f}")
    print(f"  test:  {metrics['test_accuracy']:.4f}")
    print(f"  train-test gap: {metrics['train_accuracy'] - metrics['test_accuracy']:+.4f}")

    print("\nTest per-class metrics:")
    for cls, m in metrics["test_per_class"].items():
        print(f"  {cls:6s} precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"f1={m['f1']:.3f} support={m['support']}")

    print(f"\nTop-{args.top_k} features by mean |SHAP| (test set):")
    for f, v in topk:
        print(f"  {f:24s} {v:.4f}")


if __name__ == "__main__":
    main()
