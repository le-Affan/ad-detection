"""Final audit of the explainability branch (any env; stdlib + pandas).

Checks:
  1. frozen/MANIFEST.sha256 verifies (109 files) and freeze.md / frozen/ are unchanged against fusion-v3;
  2. the pre-registration commit precedes every commit that adds results/, figures/ or the atlas;
  3. no subject-level file is tracked (cache/, subject_level/, demo/card_data/, demo/output/);
  4. committed result tables have no subject-identifier column;
  5. no test metric: no results file mentions a test split or test metrics;
  6. no new code writes to analysis/fusion_v3/ or fits a frozen artefact (grep; the only .fit is the
     in-memory MRI refit copy, P2-R4).
Exit code 0 only if every check passes.
"""
import glob
import os
import re
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402


def git(*a):
    return subprocess.run(["git", "-C", C.ROOT, *a], capture_output=True, text=True).stdout


def main():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")

    check("frozen manifest verifies", C.verify_manifest() == 109, "109 files")
    diff = git("diff", "--name-only", "fusion-v3", "--", "analysis/fusion_v3/freeze.md", "analysis/fusion_v3/frozen/")
    check("freeze.md and frozen/ unchanged vs fusion-v3", diff.strip() == "", diff.strip())
    log = git("log", "--reverse", "--format=%H", "fusion-v3..HEAD").split()
    prereg = git("log", "--reverse", "--format=%H", "fusion-v3..HEAD", "--",
                 "analysis/explainability/preregistration.md").split()
    first_prereg = log.index(prereg[0]) if prereg else 10 ** 9
    gated = git("log", "--reverse", "--format=%H", "fusion-v3..HEAD", "--", "analysis/explainability/results",
                "analysis/explainability/figures", "analysis/explainability/atlas").split()
    check("pre-registration committed before any results / figures / atlas",
          all(log.index(c) > first_prereg for c in gated), f"prereg {prereg[0][:7] if prereg else None}")
    tracked = git("ls-files").splitlines()
    bad = [f for f in tracked if re.match(r"(analysis/explainability/(cache|subject_level)/|demo/card_data/|demo/output/)", f)
           and not f.endswith(".py")]
    check("no subject-level file tracked", not bad, ", ".join(bad[:5]))
    ids = []
    for f in glob.glob(os.path.join(C.RESULTS, "*.csv")):
        cols = set(pd.read_csv(f, nrows=1).columns)
        if cols & C.FORBIDDEN_COLS:
            ids.append(os.path.basename(f))
    check("no identifier columns in results tables", not ids, ", ".join(ids))
    test_hits = []
    for f in glob.glob(os.path.join(C.RESULTS, "*")):
        if f.endswith(".lock"):
            continue
        txt = open(f, encoding="utf-8", errors="ignore").read()
        if re.search(r'"test"|\btest_|split.{0,4}test|,test,', txt):
            test_hits.append(os.path.basename(f))
    check("no test split in any results file", not test_hits, ", ".join(test_hits))
    code = glob.glob(os.path.join(HERE, "*.py")) + glob.glob(os.path.join(C.ROOT, "src", "card", "*.py"))
    writes = []
    for f in code:
        for i, line in enumerate(open(f), 1):
            if re.search(r"torch\.save|save_model|pickle\.dump", line):
                writes.append(f"{os.path.basename(f)}:{i}")
            if re.search(r"\.fit\(", line) and "in-memory copy" not in line and "fit(samp)" not in line:
                writes.append(f"{os.path.basename(f)}:{i} (.fit)")
    check("no frozen artefact written or refitted by new code", not writes, ", ".join(writes))
    print("AUDIT", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
