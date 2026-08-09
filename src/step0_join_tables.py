"""
Step 0 — Table joining.

The ADNI portal no longer distributes a single flat ADNIMERGE.csv. Its
replacement (the ADNIMERGE2 R data package) splits the same information
across ~200 domain tables. This step reconstructs the pieces our pipeline
needs:

  1. Derive DX_bl (CN / SMC / EMCI / LMCI / AD / MCI) per subject by
     combining ARM.csv (enrollment arm — carries the EMCI/LMCI/SMC
     distinction that only exists at enrollment time) with DXSUM.csv
     (baseline clinical diagnosis — the source of truth for whether that
     enrollment category actually held at the baseline visit).
  2. Reduce every visit-level source table to its own baseline row per
     subject, then join those per-subject extracts on RID.

Output is a single RID-keyed table (one row per subject) for step1 to
cohort-filter and label.

Join strategy
-------------
Visit-level ADNI tables do NOT agree on how they code a visit. Two
separate problems:

  * VISCODE is protocol-specific and disagrees across tables even when
    VISCODE2 agrees (UCSFFSX51 has VISCODE == VISCODE2 in only 0.2% of
    rows). Joining on VISCODE guarantees misalignment, so it is never
    part of a join key here.
  * Each table codes its baseline-equivalent visit differently:
    ADAS/ECOGPT/ECOGSP use 'bl', CDR uses 'sc' (screening), and
    UCSFFSX51 uses 'scmri' (screening MRI) and has zero 'bl' rows.
    Filtering a merged frame on VISCODE2 == 'bl' therefore silently
    drops CDR and MRI entirely.

So each source is collapsed to one row per RID *before* joining, using
that table's own baseline code (with each subject's earliest available
visit in that table as fallback). Once every source is one-row-per-
subject there is no visit code left to reconcile, and the joins key on
RID alone.

Column collisions are resolved explicitly (see merge_sources): any name
appearing in more than one source is suffixed with its source tag before
merging, rather than relying on pandas' merge `suffixes=` and column
ordering. A named set of canonical columns is then coalesced back to
clean names in a documented priority order.
"""
import argparse
import os

import pandas as pd

NA_VALUES = ["NA", ""]


def resolve_arm_baseline(arm_path):
    """One ARM row per RID: prefer the row where ORIGPROT == COLPROT
    (the true original-enrollment record), since re-collection rows under
    a later COLPROT can carry a stale/duplicate ARM value."""
    arm = pd.read_csv(arm_path, na_values=NA_VALUES, dtype=str)
    arm["_is_orig"] = arm["ORIGPROT"] == arm["COLPROT"]
    arm = arm.sort_values("_is_orig", ascending=False)
    arm_bl = arm.drop_duplicates(subset="RID", keep="first")
    return arm_bl.set_index("RID")["ARM"]


def arm_to_category(arm_value):
    if pd.isna(arm_value):
        return None
    if "EMCI" in arm_value:
        return "EMCI"
    if "LMCI" in arm_value:
        return "LMCI"
    if "SMC" in arm_value:
        return "SMC"
    if arm_value.startswith("AD"):
        return "AD"
    if arm_value.startswith("NL"):
        return "CN"
    if arm_value.startswith("MCI"):
        return "MCI"  # ADNI1 plain enrollment, no early/late split
    return "OTHER"


def resolve_dxsum_baseline(dxsum_path):
    """One DXSUM row per RID: VISCODE2 == 'bl' if present, else the
    earliest dated record."""
    dxsum = pd.read_csv(dxsum_path, na_values=NA_VALUES, dtype=str)
    dxsum["EXAMDATE"] = pd.to_datetime(dxsum["EXAMDATE"], errors="coerce")

    bl = dxsum[dxsum["VISCODE2"] == "bl"].drop_duplicates(subset="RID", keep="first")
    remaining = dxsum[~dxsum["RID"].isin(bl["RID"])].dropna(subset=["EXAMDATE"])
    earliest = (
        remaining.sort_values("EXAMDATE")
        .drop_duplicates(subset="RID", keep="first")
    )
    dxsum_bl = pd.concat([bl, earliest], ignore_index=True)
    return dxsum_bl.set_index("RID")["DIAGNOSIS"]


def diagnosis_to_category(diagnosis):
    if pd.isna(diagnosis):
        return None
    return {"CN": "CN", "MCI": "MCI", "Dementia": "AD"}.get(diagnosis)


def combine_dx_bl(arm_cat, dx_cat):
    """DXSUM's confirmed baseline diagnosis wins whenever it disagrees
    with the enrollment arm; the arm supplies finer-grained categories
    that DXSUM's 3-bucket (CN/MCI/Dementia) scheme can't express on its
    own: EMCI/LMCI when DXSUM confirms MCI, and SMC when DXSUM confirms
    CN (SMC subjects are, by enrollment design, objectively normal with
    only a subjective complaint, so DXSUM legitimately reads CN for
    them — collapsing that to plain CN would silently erase the SMC
    category)."""
    if dx_cat is None:
        return arm_cat
    if dx_cat == "AD":
        return "AD"
    if dx_cat == "MCI":
        if arm_cat in ("EMCI", "LMCI", "MCI"):
            return arm_cat
        return "MCI"  # dx confirms MCI but arm had no early/late signal
    if dx_cat == "CN":
        if arm_cat == "SMC":
            return "SMC"
        return "CN"
    return arm_cat


def build_dx_bl(raw_dir):
    arm_tables_dir = os.path.join(raw_dir, "adnimerge2_tables")
    arm_series = resolve_arm_baseline(os.path.join(arm_tables_dir, "ARM.csv"))
    dxsum_series = resolve_dxsum_baseline(os.path.join(arm_tables_dir, "DXSUM.csv"))

    all_rids = arm_series.index.union(dxsum_series.index)
    dx_bl = pd.DataFrame(index=all_rids)
    dx_bl["ARM_CAT"] = arm_series.reindex(all_rids).apply(arm_to_category)
    dx_bl["DXSUM_CAT"] = dxsum_series.reindex(all_rids).apply(diagnosis_to_category)
    dx_bl["DX_bl"] = dx_bl.apply(
        lambda row: combine_dx_bl(row["ARM_CAT"], row["DXSUM_CAT"]), axis=1
    )
    dx_bl.index.name = "RID"
    return dx_bl.reset_index()


# ---------------------------------------------------------------------
# Per-source baseline extraction
# ---------------------------------------------------------------------

# Each visit-level table's own baseline code, in preference order, plus
# the date column used to break ties and to drive the earliest-visit
# fallback when none of the preferred codes exist for a subject.
BASELINE_SPEC = {
    "CDR": {"codes": ("bl", "sc"), "date": "VISDATE"},
    "ADAS": {"codes": ("bl",), "date": "VISDATE"},
    "ECOGPT": {"codes": ("bl",), "date": "VISDATE"},
    "ECOGSP": {"codes": ("bl",), "date": "VISDATE"},
    "UCSFFSX51": {"codes": ("bl", "scmri"), "date": "EXAMDATE"},
    "MOCA": {"codes": ("bl",), "date": "VISDATE"},
}


# ---------------------------------------------------------------------
# ECog domain / total scores
# ---------------------------------------------------------------------

# ECOGPT.csv and ECOGSP.csv ship the derived domain and total scores, but
# they are populated ONLY for COLPROT == ADNI3 rows (0 of 4,736 ADNI2 and
# 0 of 478 ADNIGO rows carry them). Our ADNIGO/ADNI2 cohort therefore
# gets nothing from those columns and the scores must be recomputed from
# the raw items.
#
# The construction below was not assumed — it was reverse-engineered
# against the ~2,880 COLPROT == ADNI3 rows per instrument where both the
# raw items and ADNI's official derived scores are present, and it
# reproduces every official value exactly (100% value match AND 100%
# NA-pattern match on all 6 domains and the total, for both ECOGPT and
# ECOGSP). Established rules:
#   * items are stored as label strings ("2- Questionable/occasionally
#     worse"); the leading digit is the score
#   * code 9 ("I don't know") recodes to NA
#   * Visspat is a SEVEN-item domain here: VISSPAT8 is 100% empty in this
#     export. Treating it as 8 items breaks the NA pattern (99.1% / 98.3%)
#     because rows with 4 of 8 "missing" are really complete 7-item rows.
#   * a score is the mean of its items when fewer than half are missing,
#     else NA — the same rule applies at domain and total level
#   * the TOTAL is the mean of all 39 items, NOT the mean of the 6 domain
#     scores. Mean-of-domain-means matches only 3.9% (Pt) / 15.9% (SP) of
#     official values, because the domains have unequal item counts.
ECOG_DOMAINS = {
    "Mem": [f"MEMORY{i}" for i in range(1, 9)],
    "Lang": [f"LANG{i}" for i in range(1, 10)],
    "Visspat": [f"VISSPAT{i}" for i in range(1, 8)],
    "Plan": [f"PLAN{i}" for i in range(1, 6)],
    "Organ": [f"ORGAN{i}" for i in range(1, 7)],
    "Divatt": [f"DIVATT{i}" for i in range(1, 5)],
}
ECOG_ITEMS = [c for items in ECOG_DOMAINS.values() for c in items]


def _mean_if_mostly_answered(frame):
    """Mean across columns, but NA when half or more of them are missing."""
    n = frame.shape[1]
    return frame.mean(axis=1).where(frame.isna().sum(axis=1) < n / 2)


def derive_ecog_scores(df, prefix):
    """Add the 6 ECog domain scores and the total for one instrument.

    `prefix` is 'EcogPt' (self report) or 'EcogSP' (study-partner report).
    """
    items = pd.DataFrame(index=df.index)
    for col in ECOG_ITEMS:
        score = pd.to_numeric(
            df[col].astype(str).str.extract(r"^\s*(\d)")[0], errors="coerce"
        )
        items[col] = score.where(score != 9)  # 9 = "I don't know" -> NA

    for domain, cols in ECOG_DOMAINS.items():
        df[f"{prefix}{domain}"] = _mean_if_mostly_answered(items[cols])
    df[f"{prefix}Total"] = _mean_if_mostly_answered(items)
    return df


def load_ecog_table(path, prefix):
    df = load_domain_table(path)
    return derive_ecog_scores(df, prefix)


# ---------------------------------------------------------------------
# MoCA total
# ---------------------------------------------------------------------

# MOCA.csv has the same problem as the ECog tables: the summary `MOCA`
# column is populated only for COLPROT ADNI3/ADNI4 (0 of 4,629 ADNI2 and
# 0 of 477 ADNIGO rows), while the raw items are fully populated for our
# cohort (~5,050 rows). So the total is recomputed from items.
#
# As with ECog, the scoring was reverse-engineered rather than assumed,
# against the 3,706 rows carrying ADNI's official MOCA value: the formula
# below reproduces it exactly (100.00% match, mean difference 0.0000, on
# all 3,703 rows where both are computable). Notes:
#   * ADNI stores the RAW score with no education adjustment (the usual
#     "+1 if <=12 years education" rule is not applied — adding it drops
#     agreement to 0%)
#   * IMMT1W*/IMMT2W* (immediate recall trials) are not scored
#   * delayed recall counts only "Correct with No Cue"
MOCA_ONE_POINT = [
    "TRAILS", "CUBE", "CLOCKCON", "CLOCKNO", "CLOCKHAN",   # visuospatial/exec
    "LION", "RHINO", "CAMEL",                              # naming
    "DIGFOR", "DIGBACK",                                   # attention
    "REPEAT1", "REPEAT2",                                  # language
    "ABSTRAN", "ABSMEAS",                                  # abstraction
    "DATE", "MONTH", "YEAR", "DAY", "PLACE", "CITY",       # orientation
]
MOCA_SERIAL = [f"SERIAL{i}" for i in range(1, 6)]
MOCA_DELAYED = [f"DELW{i}" for i in range(1, 6)]


def derive_moca_total(df):
    parts = [(df[c] == "Correct").astype(float).where(df[c].notna())
             for c in MOCA_ONE_POINT]

    letters = pd.to_numeric(df["LETTERS"], errors="coerce")
    parts.append((letters <= 1).astype(float).where(letters.notna()))

    n_serial = sum((df[c] == "Correct").astype(float) for c in MOCA_SERIAL)
    parts.append(
        pd.cut(n_serial, [-0.1, 0.1, 1.1, 3.1, 5.1], labels=[0, 1, 2, 3]).astype(float)
    )

    fluency = pd.to_numeric(df["FFLUENCY"], errors="coerce")
    parts.append((fluency >= 11).astype(float).where(fluency.notna()))

    parts.append(
        sum((df[c] == "Correct with No Cue").astype(float) for c in MOCA_DELAYED)
    )

    # every component must be scorable, otherwise the total is not comparable
    stacked = pd.concat(parts, axis=1)
    df["MOCA"] = stacked.sum(axis=1).where(stacked.notna().all(axis=1))
    return df


def load_moca_table(path):
    return derive_moca_total(load_domain_table(path))


def derive_apoe4(genotype):
    """APOE4 = number of e4 alleles (0/1/2), from ADSL's genotype string
    (e.g. 'e3/e4'). The doc's feature list asks for the allele count, not
    the genotype label; the raw genotype is kept alongside as a
    categorical."""
    if pd.isna(genotype):
        return None
    return str(genotype).count("4")


def select_baseline_row(df, codes, date_col):
    """Collapse a visit-level table to one row per RID.

    Rows are ranked by which preferred baseline code they carry (tier 0 =
    first code, tier 1 = second, ...), with every other visit falling
    into a final fallback tier. Within a tier the earliest dated visit
    wins. So a subject with the expected baseline code always yields that
    row, and a subject missing it falls back to their earliest available
    visit in this table.
    """
    df = df.copy()
    tier = pd.Series(len(codes), index=df.index, dtype=int)
    for i, code in enumerate(codes):
        tier = tier.mask(df["VISCODE2"] == code, i)
    df["_tier"] = tier
    df["_date"] = pd.to_datetime(df.get(date_col), errors="coerce")

    df = df.sort_values(["_tier", "_date"], na_position="last")
    df = df.drop_duplicates(subset="RID", keep="first")
    return df.drop(columns=["_tier", "_date"])


def load_domain_table(path):
    df = pd.read_csv(path, na_values=NA_VALUES, dtype=str, low_memory=False)
    if "RID" not in df.columns and "SUBJID" in df.columns:
        # ADSL and other ADaM-style tables key subjects by SUBJID, not RID.
        df["RID"] = df["SUBJID"]
    df["RID"] = df["RID"].astype(str)
    return df


# UCSFFSX51 uses raw FreeSurfer ROI codes, not the human-readable region
# names classic ADNIMERGE exposed. Each region is the plain SUM of the
# left and right hemisphere volumes, matching ADNIMERGE's own
# construction code (adnimerger.r):
#     ucsfvol$Hippocampus <- (ucsfvol$ST29SV + ucsfvol$ST88SV)
# ADNI's "Commonly Used Tables" PDF shows (ST29SV+ST88SV)/2, but that
# describes ad-hoc use of the older UCSFFSX.csv (FreeSurfer 4.3, ADNI1
# 1.5T) and is out of sync with the shipped ADNIMERGE build. Sums are
# what ADNIMERGE actually distributes, and the resulting magnitudes match
# published ADNIMERGE values (Hippocampus ~7,000 mm^3); halving them
# matches nothing.
ROI_PAIRS = {
    "Ventricles": ("ST37SV", "ST96SV"),
    "Hippocampus": ("ST29SV", "ST88SV"),
    "Entorhinal": ("ST24CV", "ST83CV"),
    "Fusiform": ("ST26CV", "ST85CV"),
    "MidTemp": ("ST40CV", "ST99CV"),
}
ROI_SINGLE = {
    "ICV": "ST10CV",
}

# WholeBrain is the sum of ADNI's documented per-lobe/subcortical volume
# codes. ADNI documents the formula against the FreeSurfer 4.3 table
# (UCSFFSX.csv) and separately instructs that, for the 5.1 table
# (UCSFFSX51.csv), the four version-4.3 codes ST19SV/ST20SV/ST78SV/ST79SV
# are REPLACED by their 5.1 equivalents:
#     ST19SV LeftCerebralCortex  -> ST147SV LeftCorticalGM
#     ST78SV RightCerebralCortex -> ST148SV RightCorticalGM
#     ST20SV LeftCerebralWM      -> ST150SV LeftCorticalWM
#     ST79SV RightCerebralWM     -> ST151SV RightCorticalWM
# All four 5.1 codes are present in this export, so the formula below is
# complete (25 codes, same count as the documented 4.3 formula) and the
# resulting mean (~1.04e6 mm^3) matches published ADNIMERGE WholeBrain.
WHOLEBRAIN_CODES = [
    "ST128SV", "ST17SV", "ST18SV", "ST61SV", "ST16SV", "ST53SV", "ST42SV",
    "ST29SV", "ST12SV", "ST11SV", "ST65SV", "ST76SV", "ST77SV", "ST120SV",
    "ST75SV", "ST112SV", "ST101SV", "ST88SV", "ST71SV", "ST70SV", "ST124SV",
    "ST147SV", "ST148SV", "ST150SV", "ST151SV",
]


def load_mri_volumetrics(path):
    keep_cols = ["RID", "VISCODE", "VISCODE2", "EXAMDATE", "ORIGPROT"]
    header = pd.read_csv(path, nrows=0).columns.tolist()

    missing = [c for c in WHOLEBRAIN_CODES if c not in header]
    if missing:
        raise ValueError(
            f"UCSFFSX51 is missing expected WholeBrain codes {missing}. "
            f"WholeBrain cannot be computed correctly; check the export."
        )

    all_roi_cols = sorted(
        set(
            [c for pair in ROI_PAIRS.values() for c in pair]
            + list(ROI_SINGLE.values())
            + WHOLEBRAIN_CODES
        )
    )
    df = pd.read_csv(
        path, na_values=NA_VALUES, dtype=str, low_memory=False,
        usecols=keep_cols + all_roi_cols,
    )
    df["RID"] = df["RID"].astype(str)

    for col in all_roi_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for region, (left, right) in ROI_PAIRS.items():
        df[region] = df[left] + df[right]
    for region, col in ROI_SINGLE.items():
        df[region] = df[col]
    df["WholeBrain"] = df[WHOLEBRAIN_CODES].sum(axis=1, min_count=1)

    out_cols = keep_cols + list(ROI_PAIRS) + list(ROI_SINGLE) + ["WholeBrain"]
    return df[out_cols]


# ---------------------------------------------------------------------
# Explicit collision handling
# ---------------------------------------------------------------------

# Columns that legitimately exist in several sources and should end up
# with one clean name, listed in the order the value should be taken
# from. ADSL is subject-level and authoritative for enrollment and
# summary-score fields, so it leads wherever it appears — this is what
# keeps ORIGPROT from resolving to whichever visit table happened to be
# merged first, and what keeps the populated ADSL CDRSB/CDGLOBAL from
# being shadowed by the empty CDR-sourced copies.
#
# Deliberately NOT coalesced: the ECOG item columns (MEMORY*, LANG*,
# PLAN*, ORGAN*, DIVATT*, VISSPAT*), which mean different things in
# ECOGPT (self report) vs ECOGSP (study-partner report), and the
# per-source visit/administrative columns, which are kept suffixed as
# provenance for the baseline row each source contributed.
CANONICAL_PRIORITY = {
    "ORIGPROT": ["ADSL", "ECOGPT", "ECOGSP", "UCSFFSX51", "MOCA"],
    "CDRSB": ["ADSL", "CDR"],
    "CDGLOBAL": ["ADSL", "CDR"],
    # ADSL's MOCA copy is empty for this cohort; the dedicated MOCA domain
    # table leads, with ADSL only filling gaps.
    "MOCA": ["MOCA", "ADSL"],
    "SITEID": ["ADSL", "CDR", "ADAS", "ECOGPT", "ECOGSP"],
    "PTID": ["CDR", "ADAS", "ECOGPT", "ECOGSP", "MOCA"],
    "PHASE": ["CDR", "ADAS"],
}


def merge_sources(sources):
    """Merge {tag: one-row-per-RID frame} on RID with explicit collision
    handling.

    Any column name present in more than one source is renamed to
    ``NAME__TAG`` before merging, so no value can be silently shadowed by
    merge ordering. Columns unique to a single source keep their clean
    name. Canonical columns are then coalesced back to a clean name using
    CANONICAL_PRIORITY, and the provenance of each is reported.
    """
    counts = {}
    for tag, df in sources.items():
        for col in df.columns:
            if col == "RID":
                continue
            counts[col] = counts.get(col, 0) + 1
    collided = {c for c, n in counts.items() if n > 1}

    renamed = {}
    for tag, df in sources.items():
        mapping = {c: f"{c}__{tag}" for c in df.columns if c in collided}
        renamed[tag] = df.rename(columns=mapping)

    merged = None
    for tag, df in renamed.items():
        merged = df if merged is None else merged.merge(df, on="RID", how="outer")

    provenance = []
    for canon, priority in CANONICAL_PRIORITY.items():
        candidates = [f"{canon}__{t}" for t in priority if f"{canon}__{t}" in merged.columns]
        if canon in merged.columns:  # never collided, already clean
            provenance.append((canon, "(uncollided)", int(merged[canon].notna().sum())))
            continue
        if not candidates:
            continue
        out = merged[candidates[0]]
        for c in candidates[1:]:
            out = out.combine_first(merged[c])
        merged[canon] = out
        # report which source supplied the value for each row
        winner = candidates[0].split("__", 1)[1]
        provenance.append((canon, winner, int(merged[canon].notna().sum())))

    return merged, provenance, collided


def join_tables(raw_dir, out_path):
    tables_dir = os.path.join(raw_dir, "adnimerge2_tables")

    dx_bl = build_dx_bl(raw_dir)
    dx_bl["RID"] = dx_bl["RID"].astype(str)

    raw_sources = {
        "CDR": load_domain_table(os.path.join(raw_dir, "CDR.csv")),
        "ADAS": load_domain_table(os.path.join(raw_dir, "ADAS_ADNIGO2.csv")),
        "ECOGPT": load_ecog_table(os.path.join(tables_dir, "ECOGPT.csv"), "EcogPt"),
        "ECOGSP": load_ecog_table(os.path.join(tables_dir, "ECOGSP.csv"), "EcogSP"),
        "UCSFFSX51": load_mri_volumetrics(os.path.join(tables_dir, "UCSFFSX51.csv")),
        # MOCA has its own domain table; ADSL's MOCA copy is empty for the
        # ADNIGO/ADNI2 cohort, so the dedicated table is the real source.
        "MOCA": load_moca_table(os.path.join(tables_dir, "MOCA.csv")),
    }

    # Collapse every visit-level source to its own baseline row per RID
    # BEFORE any joining, so the merges below key on RID alone.
    baseline_sources = {}
    print("Per-source baseline extraction:")
    for tag, df in raw_sources.items():
        spec = BASELINE_SPEC[tag]
        n_before = len(df)
        bl = select_baseline_row(df, spec["codes"], spec["date"])
        hit = bl["VISCODE2"].isin(spec["codes"]).sum()
        print(
            f"  {tag:10s} {n_before:6d} visit rows -> {len(bl):5d} subjects "
            f"({hit} via {list(spec['codes'])}, {len(bl) - hit} via earliest-visit fallback)"
        )
        baseline_sources[tag] = bl

    # ADSL is already subject-level; DX_bl is derived per subject.
    baseline_sources["ADSL"] = load_domain_table(os.path.join(tables_dir, "ADSL.csv"))
    baseline_sources["DX"] = dx_bl

    joined, provenance, collided = merge_sources(baseline_sources)
    joined["APOE4"] = joined["APOE"].apply(derive_apoe4)

    print(f"\nCollision handling: {len(collided)} column names appeared in >1 source "
          f"and were suffixed with their source tag.")
    print("Canonical columns coalesced (priority order applied):")
    for canon, winner, nn in provenance:
        print(f"  {canon:10s} <- {winner:12s} non-null {nn}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joined.to_csv(out_path, index=False)
    return joined


def main():
    parser = argparse.ArgumentParser(description="Step 0: join ADNI tables")
    parser.add_argument("--raw-dir", default="dataset/raw")
    parser.add_argument("--out", default="dataset/processed/step0_joined.csv")
    args = parser.parse_args()

    joined = join_tables(args.raw_dir, args.out)
    print(f"\nJoined table: {joined.shape[0]} rows, {joined.shape[1]} cols -> {args.out}")
    print(f"Unique RIDs: {joined['RID'].nunique()}")

    mri_cols = list(ROI_PAIRS) + list(ROI_SINGLE) + ["WholeBrain"]
    print("\nMRI volumetric coverage (all subjects):")
    for c in mri_cols:
        vals = pd.to_numeric(joined[c], errors="coerce")
        print(f"  {c:12s} non-null {vals.notna().sum():5d}  mean {vals.mean():12,.0f}")

    print()
    print(joined["DX_bl"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
