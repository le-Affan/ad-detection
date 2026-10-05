"""Exact linear SHAP for the frozen MRI specialist (LYP/.venv-fusion; numpy only at call time).

The specialist (analysis/fusion_v3/frozen/mri_<seed>.pkl, specialists.MRI kind "logreg") is a multinomial
logistic regression on z = (impute_median(x) - mu) / sd, x = [6 volume/ICV ratios, ALZ, BAG, AGE, SEX].
Explained target: the centred class scores s_k(z) - mean_k s_k(z), which are exactly the centred
log-probabilities the KAN receives (up to the 1e-4 clip). With an independent (interventional) background B,
  phi_kj = Wc_kj * (z_j - mean_B z_j),  Wc = W - mean_k W,  base_k = Wc_k . mean_B z + bc_k.
Imputed inputs are explained at their imputed value and flagged.
"""
import numpy as np

FEATURES = ["Ventricles_ICV", "Hippocampus_ICV", "WholeBrain_ICV", "Entorhinal_ICV", "Fusiform_ICV",
            "MidTemp_ICV", "ALZ", "BAG", "AGE", "SEX"]
LABELS = {"Ventricles_ICV": "Ventricles", "Hippocampus_ICV": "Hippocampus", "WholeBrain_ICV": "Whole brain",
          "Entorhinal_ICV": "Entorhinal cortex", "Fusiform_ICV": "Fusiform gyrus",
          "MidTemp_ICV": "Middle temporal gyrus", "ALZ": "CNN AD score (ALZ)", "BAG": "Brain-age gap",
          "AGE": "Age", "SEX": "Sex"}
GROUPS = {"volumes": FEATURES[:6], "ALZ+BAG": ["ALZ", "BAG"], "age+sex": ["AGE", "SEX"]}


def design(mri, df):
    """Rows with a scan: (z, missing mask) in the specialist's own preprocessing."""
    x = mri._X(df)
    miss = np.isnan(x)
    z = (np.where(miss, mri.med, x) - mri.mu) / mri.sd
    return z, miss


def centred_coef(mri):
    W, b = mri.model.coef_, mri.model.intercept_
    return W - W.mean(0, keepdims=True), b - b.mean()


def shap_values(mri, z, z_bg):
    """phi (n, 10, 4) on centred scores, base (4,)."""
    Wc, bc = centred_coef(mri)
    mu = z_bg.mean(0)
    phi = (z - mu)[:, :, None] * Wc.T[None]
    base = Wc @ mu + bc
    return phi, base


def centred_scores(mri, z):
    s = mri.model.decision_function(z)
    return s - s.mean(1, keepdims=True)


def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)
