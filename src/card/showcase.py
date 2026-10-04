"""Showcase subjects for the assessment card: a fixed rule, written and committed
before any card output was generated or looked at.

Rule. From the VAL split, ordered by subject ID (PTID, ascending string order),
take the first subject in each category, in this order:
  1. correct CN     true CN,   final (KAN) prediction CN
  2. correct EMCI   true EMCI, final prediction EMCI
  3. correct LMCI   true LMCI, final prediction LMCI
  4. correct AD     true AD,   final prediction AD
  5. misclassified LMCI   true LMCI, final prediction not LMCI (errors are shown, not hidden)
  6. discordance flag     the exploratory clinical-vs-imaging flag fires
  7. unreliable brain-age scan   the scan fails the pre-registered reliability rule
  8. no scan              the subject has no MRI scan (missing-MRI behaviour)
A subject already chosen for an earlier category is skipped, so the set has 8
distinct subjects. If a category has no val subject, the same rule is applied
to the TRAIN split for that category, and the card says so. (The kan-fusion
reliability audit found no val scan failing the rule, so category 7 is
expected to fall back to train.)
"Final prediction" is the frozen KAN fusion's argmax. Val and train subjects are
in the frozen system's training data (fitted on all 691 train+val subjects), so
their predictions are in-sample; every card says this.
"""
import pandas as pd

CATEGORIES = [
    ("correct CN", lambda d: (d.true == "CN") & (d.pred == "CN")),
    ("correct EMCI", lambda d: (d.true == "EMCI") & (d.pred == "EMCI")),
    ("correct LMCI", lambda d: (d.true == "LMCI") & (d.pred == "LMCI")),
    ("correct AD", lambda d: (d.true == "AD") & (d.pred == "AD")),
    ("misclassified LMCI", lambda d: (d.true == "LMCI") & (d.pred != "LMCI")),
    ("discordance flag", lambda d: d.discordant == True),                          # noqa: E712
    ("unreliable brain-age scan", lambda d: d.ba_unreliable == True),              # noqa: E712
    ("no scan", lambda d: d.has_mri == False),                                     # noqa: E712
]


def select(subjects: pd.DataFrame) -> pd.DataFrame:
    """subjects: one row per subject with PTID, split, true, pred, discordant,
    ba_unreliable, has_mri. Returns category, PTID, split, fallback."""
    chosen, rows = set(), []
    for name, rule in CATEGORIES:
        pick = None
        for split in ("val", "train"):
            d = subjects[subjects.split == split].sort_values("PTID")
            hit = d[rule(d) & ~d.PTID.isin(chosen)]
            if len(hit):
                pick = (hit.PTID.iloc[0], split)
                break
        if pick is None:
            rows.append({"category": name, "PTID": None, "split": None, "fallback": None})
            continue
        chosen.add(pick[0])
        rows.append({"category": name, "PTID": pick[0], "split": pick[1], "fallback": pick[1] != "val"})
    return pd.DataFrame(rows)
