"""Write the sha256 manifest of the frozen artefacts and code (frozen/MANIFEST.sha256).

freeze.md cites this manifest; test scripts are run only after both are committed.
"""
import glob
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

CODE = ["metrics.py", "specialists.py", "kan_fusion.py", "final_features.py", "freeze_guard.py",
        "freeze_fit_specialists.py", "freeze_fit_kan.py", "scan_inputs.py", "brainage_split.py",
        "test_streams.py", "test_kan.py", "test_evaluate.py"]
EXTERNAL = ["src/step4_processing.py", "analysis/fusion_feasibility/predict_alz.py",
            "analysis/brain_age/compute_brainage.py", "analysis/brain_age/dbn_smoke.py",
            "analysis/fusion_feasibility/alz_crossfit.csv", "analysis/brain_age/brainage_trainval.csv",
            "analysis/kan_fusion/brainage_reliability.csv", "analysis/kan_fusion/topk.json"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    files = sorted(glob.glob(os.path.join(HERE, "frozen", "*")))
    files = [f for f in files if not f.endswith("MANIFEST.sha256")]
    files += [os.path.join(HERE, c) for c in CODE] + [os.path.join(ROOT, e) for e in EXTERNAL]
    files += sorted(glob.glob(os.path.join(ROOT, "analysis", "fusion_feasibility", "runs", "fold*_s*", "best.pt")))
    lines = [f"{sha(f)}  {os.path.relpath(f, ROOT)}" for f in files]
    with open(os.path.join(HERE, "frozen", "MANIFEST.sha256"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{len(lines)} files hashed")
    print(sha(os.path.join(HERE, "frozen", "MANIFEST.sha256")), "MANIFEST.sha256")


if __name__ == "__main__":
    main()
