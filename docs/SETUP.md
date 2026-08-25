# Setting up on a new machine

A clone gives you the **entire clinical stream** — raw ADNI tables, every
intermediate, the trained model — so `preprocessing.py` runs end to end on a
fresh machine with nothing else installed. The MRI stream is the one that still
needs an out-of-band transfer: it clones this repo, drops one zip in one place,
and runs from there.

---

## 1. What the clone gives you, and what it does not

Tracked, so you get it from the clone:

    src/                          all pipeline source (clinical + MRI)
    preprocessing.py              clinical master runner
    requirements.txt              clinical deps (Python 3.14)
    requirements-mri.txt          MRI deps (Python 3.12)
    dataset/mri_manifest/         mri_manifest.csv - 814 subjects, PTID/RID/label
    dataset/mri_raw/*.py          unpack + scan-selection scripts
    dataset/mri_raw/selected_scans.csv     the 811-scan selection, full provenance
    docs/                         spec and session handoffs

    dataset/raw/                  752 MB of raw ADNI tables (~115 MB packed)
    dataset/processed/            every clinical intermediate, step0 -> step4
    dataset/processed/*_rids.csv  the train/val/test split - see note below
    models/                       trained CatBoost model, metrics, SHAP artefacts
    catboost_info/                training curves from the step5 run

NOT tracked (too large for git - transfer these out of band):

    dataset/mri_raw/<PTID>/       extracted NIfTI, ~34 GB
    dataset/mri_prep/             preprocessed volumes, ~4.6 GB
    dataset/*.tar                 outbound MRI bundles

**The split files are tracked on purpose.** `dataset/processed/{train,val,test}_rids.csv`
are 4 KB total and are the one clinical artefact the MRI stream cannot work
without: `m2_index.py` reads them to assign each scan a split. They are also
what keeps the two streams fusable — both must use the *same* subject-level
assignment or the fusion model leaks across splits. Everything else in
`dataset/processed/` is rebuildable from `dataset/raw/`, which this machine
will not have.

**The clinical pipeline needs nothing but the clone.** `dataset/raw/` is
tracked, so `python preprocessing.py` reruns steps 0–5 on any machine with
`requirements.txt` installed. The step0–step4 outputs and the trained model are
tracked too, so you can also just read them without rerunning anything.

**This repo is private and must stay private.** `dataset/raw/` and the
`dataset/processed/` intermediates are DUA-restricted ADNI subject data. Do not
make the remote public and do not fork it anywhere public.

---

## 2. Prerequisites

- **Python 3.12** for the MRI stream. `antspyx` publishes wheels for cp38–cp313
  only; on 3.14 pip falls back to compiling ITK + ANTs from source. Any of
  3.8–3.13 works, 3.12 is what this was built and tested on.
- ~**70 GB free disk** at peak (27 GB zip + 34 GB extracted + 4.6 GB
  preprocessed). You can delete the zip after unpacking to get 27 GB back.
- Network on first run — `nilearn` downloads the ICBM152 template once
  (~50 MB) and caches it in `~/nilearn_data`.

---

## 3. Where to put the zip

```
dataset/mri_raw/_zips/<your-file>.zip
```

Create the directory if it does not exist:

```bash
mkdir -p dataset/mri_raw/_zips
mv /path/to/your/download.zip dataset/mri_raw/_zips/
```

The filename does not matter. `unpack_and_verify.py` globs `*.zip` in that
directory, and matches every extracted file back to a subject by **Image ID**
parsed from the archive path — not by filename, zip name, or ordering. So one
big zip and ten small zips both work, and a zip containing more scans than our
811 selection is fine too: extra files are reported as unmatched and skipped.

That directory is gitignored, deny-by-default. Nothing you put there can be
committed by accident.

---

## 4. Run it

```bash
git clone https://github.com/le-Affan/ad-detection.git
cd ad-detection

# MRI environment (Python 3.12)
python3.12 -m venv .venv312
.venv312/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv312/bin/pip install -r requirements-mri.txt
```

**If that machine has an NVIDIA GPU**, install torch from the default index
instead so you get CUDA:

```bash
.venv312/bin/pip install torch          # instead of the --index-url line
.venv312/bin/pip install -r requirements-mri.txt
```

then drop the `torch==2.13.0+cpu` pin from `requirements-mri.txt` before
installing, or install it after and let pip keep the CUDA build.

Then, with the zip in place:

```bash
# 1. unpack -> dataset/mri_raw/<PTID>/, writes downloaded_manifest.csv
.venv312/bin/python dataset/mri_raw/unpack_and_verify.py

# 2. preprocess -> dataset/mri_prep/{mni,vol96}/    (~5.7 h for 811)
.venv312/bin/python src/mri/m1_preprocess.py --tag full

# 3. build the index the dataset class reads
.venv312/bin/python src/mri/m2_index.py
```

Each step reads whatever is on disk at run time and skips work already done,
so any of them can be interrupted and re-run without losing progress or
duplicating effort.

### Expected output

- **step 1**: 811 subjects. Three RIDs (2360, 4095, 4187) have no baseline
  structural T1 in IDA and are excluded by design — the script lists them
  under "EXCLUDED BY DESIGN". 811, not 814, is correct.
- **step 2**: ~25 s per subject, single-threaded-ish (ANTs already uses ~10
  cores). **A GPU does not speed this up** — ANTs registration is CPU-only.
  Only training benefits from a GPU.
- **step 3**: an index of ~811 rows split roughly 569/122/123 minus the three
  missing subjects.

### Sanity check before training

```bash
.venv312/bin/python -c "
import sys; sys.path.insert(0,'src')
from mri.dataset import ADNIVolumes
for s in ('train','val','test'):
    d = ADNIVolumes(split=s); print(s, len(d), d.class_counts())
"
```

---

## 5. Before fusion, you will need one more thing

The fusion model concatenates the clinical Top-K vector with `[DEMO, ALZ]`
from the MRI stream. The clinical side of that lives in files this repo does
**not** currently track:

    dataset/processed/step4_{train,val,test}.csv   ~570 KB
    dataset/processed/step4_manifest.json          normalisation stats
    models/step5_topk_features.json                the Top-K feature list

They are small enough to track and are the same category of data as the
already-tracked manifest. They are left out for now as a deliberate
data-governance choice, not an oversight. Decide before starting fusion work,
or that machine will be blocked again.

---

## 6. Security

This repository is **private**, and that is what makes it acceptable to track
`mri_manifest.csv`, `selected_scans.csv` and the split files at all — they
carry PTIDs, diagnoses and exam dates for 814 people under the ADNI Data Use
Agreement, which prohibits redistribution. **If this repo is ever made public,
those files must be removed first.**

Never commit `dataset/mri_raw/_zips/zip_urls.txt` if it exists — those
pre-authenticated IDA links download the whole collection with no login. It is
gitignored deny-by-default and named explicitly in `.gitignore` so it cannot be
whitelisted back in by accident.
