# Alzheimer's Disease Stage Detection using CNN + KAN

> **Setting up on a new machine?** See [`docs/SETUP.md`](docs/SETUP.md) — it
> covers the two virtualenvs, where to put the MRI dataset zip, and the exact
> run order. Session handoffs are in `docs/context*.txt`.

## Overview
This project implements an interpretable deep learning framework for multi-stage Alzheimer’s disease classification and continuous severity scoring.

The system combines:
- CNN-based feature extraction
- KAN (Kolmogorov–Arnold Network) reasoning layer
- Multi-task learning (classification + regression)
- Intrinsic interpretability (no post-hoc methods)

---

## Objectives
- Classify disease stages: CN, EMCI, LMCI, AD
- Generate a continuous severity score (0–1)
- Provide feature-level explanations
- Maintain longitudinal consistency across visits

---

## Architecture
Raw Data  
→ Data Engineering  
→ Feature Engineering  
→ CNN Feature Extractor  
→ KAN Reasoning Layer  
→ Multi-Task Output  
  • Stage Classification  
  • Severity Regression  
→ Explainability Module  

---

## Project Structure

```
alzheimer-stage-detection/
│
├── configs/              # YAML configuration files
├── data/                 # Raw and processed datasets (not tracked in git)
├── notebooks/            # Research and experimentation notebooks
├── outputs/              # Models, logs, results (not tracked)
├── scripts/              # CLI scripts (train, evaluate, preprocess)
├── src/                  # Core source code
│   ├── data/             # Dataset + preprocessing modules
│   ├── models/           # CNN, KAN, multi-task models
│   ├── training/         # Training loops, losses
│   ├── evaluation/       # Metrics and statistical tests
│   ├── explainability/   # KAN interpretation logic
│   └── utils/            # Helpers (logging, config, reproducibility)
│
├── tests/                # Unit tests
├── README.md
└── requirements.txt
```




---

## Tech Stack
- Python
- PyTorch
- NumPy / Pandas
- Scikit-learn
- Matplotlib

---

## Team
- Affan Shaikh  
- Aneeket Banerjee  
- Ankon Mukherjee  
- Eshani Kane