"""Stream specialists for fusion-v3 (run with LYP/.venv-fusion). Never reads test.

Clinical specialist: CatBoost (harness settings, fixed 265 trees) on the clinical
top-K of the 29 arm-A0 features. K = 12, chosen on train CV in kan-fusion. The
ranking is refitted inside every fit: CatBoost on the fit's own rows with all
29 features, mean |SHAP| over those rows, top 12, refit. This is the
kan-fusion CatBoost C procedure.

MRI specialist: on the 6 FreeSurfer volumes / ICV, cross-fitted ALZ, the
corrected brain-age gap, age and sex. It is fitted on, and scores, subjects
with a scan only (no scan -> NaN; the fusion layer treats that stream as
missing).
  - Brain-age gap = (pred_age - beta) / alpha - age_at_scan, where (alpha, beta)
    is an OLS fit of pred_age on age_at_scan over the CN rows of the specialist's
    own training rows. So it is refitted inside every fold and nested fold.
  - The gap is NaN if the scan fails the brain-age reliability rule
    (kan_fusion/brainage_reliability.csv) or has no brain age.
  - Kinds: "catboost" (shallow, few trees), "logreg" (multinomial, L2),
    "ordinal" (Frank & Hall cumulative binary logistic), "mlp" (sklearn).
  - Imputation (medians) and scaling for logreg/ordinal/mlp are fitted on the
    training rows.
"""
import os
import sys

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "analysis", "fusion_feasibility"))
import ablation as H  # noqa: E402

RATIOS = ["Ventricles_ICV", "Hippocampus_ICV", "WholeBrain_ICV", "Entorhinal_ICV", "Fusiform_ICV", "MidTemp_ICV"]
MRI_FEATURES = RATIOS + ["ALZ", "BAG", "AGE", "SEX"]
K_CLIN = 12


def load_trainval():
    """fusion_trainval.csv + brain age + reliability flag. Train and val only."""
    d = pd.read_csv(os.path.join(ROOT, "analysis", "kan_fusion", "fusion_trainval.csv"))
    ba = pd.read_csv(os.path.join(ROOT, "analysis", "brain_age", "brainage_trainval.csv"),
                     usecols=["RID", "pred_age", "age_at_scan"])
    rel = pd.read_csv(os.path.join(ROOT, "analysis", "kan_fusion", "brainage_reliability.csv"),
                      usecols=["RID", "unreliable"])
    d = d.merge(ba, on="RID", how="left").merge(rel, on="RID", how="left")
    d["ba_ok"] = d.pred_age.notna() & (d.unreliable != True)              # noqa: E712
    d["SEX"] = (d.PTGENDER == "Male").astype(float)
    test = set(pd.read_csv(os.path.join(ROOT, "dataset", "processed", "test_rids.csv")).RID)
    assert not d.RID.isin(test).any()
    return d


def clinical_features():
    import json
    return json.load(open(os.path.join(ROOT, "analysis", "kan_fusion", "fusion_manifest.json")))["clinical_features"]


# --- clinical ---------------------------------------------------------------------
def _cb(X, y, seed, **over):
    p = {**H.PARAMS, **over}
    m = CatBoostClassifier(random_seed=seed, cat_features=[c for c in ("PTGENDER",) if c in X.columns], **p)
    m.fit(X, y)
    return m


class Clinical:
    def __init__(self, seed, k=K_CLIN):
        self.seed, self.k, self.cols = seed, k, clinical_features()

    def fit(self, df):
        X, y = df[self.cols], df.label.to_numpy()
        m = _cb(X, y, self.seed)
        sv = np.abs(np.asarray(shap.TreeExplainer(m).shap_values(X)))
        imp = sv.mean(axis=(0, 2)) if sv.ndim == 3 else sv.mean(0)
        self.top = [self.cols[i] for i in np.argsort(-imp, kind="stable")][:self.k]
        self.model = _cb(df[self.top], y, self.seed)
        return self

    def predict_proba(self, df):
        return self.model.predict_proba(df[self.top])


# --- MRI ----------------------------------------------------------------------------
class MRI:
    def __init__(self, kind, seed, **hp):
        self.kind, self.seed, self.hp = kind, seed, hp

    def _X(self, df):
        X = df[RATIOS + ["ALZ", "AGE", "SEX"]].copy()
        X["BAG"] = np.where(df.ba_ok, (df.pred_age - self.beta) / self.alpha - df.age_at_scan, np.nan)
        return X[MRI_FEATURES].to_numpy(dtype=float)

    def fit(self, df):
        df = df[df.has_mri]
        cn = df[(df.label == 0) & df.ba_ok]
        self.alpha, self.beta = np.polyfit(cn.age_at_scan, cn.pred_age, 1)
        X, y = self._X(df), df.label.to_numpy()
        if self.kind == "catboost":
            self.model = CatBoostClassifier(loss_function="MultiClass", random_seed=self.seed, verbose=False,
                                            allow_writing_files=False, **self.hp).fit(X, y)
            return self
        self.med = np.nanmedian(X, 0)
        X = np.where(np.isnan(X), self.med, X)
        self.mu, self.sd = X.mean(0), X.std(0)
        X = (X - self.mu) / self.sd
        if self.kind == "logreg":
            self.model = LogisticRegression(C=self.hp["C"], max_iter=5000).fit(X, y)
        elif self.kind == "ordinal":
            self.model = [LogisticRegression(C=self.hp["C"], max_iter=5000).fit(X, (y > k).astype(int))
                          for k in range(3)]
        elif self.kind == "mlp":
            self.model = MLPClassifier(hidden_layer_sizes=self.hp["hidden"], alpha=self.hp["alpha"],
                                       early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
                                       max_iter=2000, random_state=self.seed).fit(X, y)
        return self

    def predict_proba(self, df):
        out = np.full((len(df), 4), np.nan)
        rows = np.flatnonzero(df.has_mri.to_numpy())
        if len(rows) == 0:
            return out
        X = self._X(df.iloc[rows])
        if self.kind == "catboost":
            out[rows] = self.model.predict_proba(X)
            return out
        X = (np.where(np.isnan(X), self.med, X) - self.mu) / self.sd
        if self.kind == "ordinal":
            g = np.stack([m.predict_proba(X)[:, 1] for m in self.model], 1)       # P(y > k), k = 0..2
            g = np.minimum.accumulate(g, axis=1)                                   # enforce monotone
            p = np.stack([1 - g[:, 0], g[:, 0] - g[:, 1], g[:, 1] - g[:, 2], g[:, 2]], 1)
            p = np.clip(p, 1e-6, None)
            out[rows] = p / p.sum(1, keepdims=True)
        else:
            out[rows] = self.model.predict_proba(X)
        return out


MRI_CANDIDATES = {
    "cb_d2_t150": ("catboost", dict(depth=2, iterations=150, learning_rate=0.05, l2_leaf_reg=10)),
    "cb_d3_t300": ("catboost", dict(depth=3, iterations=300, learning_rate=0.03, l2_leaf_reg=10)),
    "logreg_C0.1": ("logreg", dict(C=0.1)),
    "logreg_C1": ("logreg", dict(C=1.0)),
    "ordinal_C0.1": ("ordinal", dict(C=0.1)),
    "mlp_16": ("mlp", dict(hidden=(16,), alpha=1e-2)),
}
