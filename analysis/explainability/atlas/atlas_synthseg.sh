#!/usr/bin/env bash
# Atlas, step 2 of 3: SynthSeg 2.0 (FreeSurfer 7.4.1 mri_synthseg) on the masked template.
# Run ALONE (peak ~11 GB RAM of 15). Measured 2026-10-05: 39 s wall, 10.9 GB peak.
# Same invocation as analysis/mri_morphometry/run_synthseg.py (--parc --cpu --crop 160 192 160).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LYP="$(cd "$HERE/../../../../.." && pwd)"
export FREESURFER_HOME="$LYP/third_party/fs_synthseg"
cd "$HERE/work"
"$LYP/.venv-synthseg/bin/python" "$FREESURFER_HOME/bin/mri_synthseg" --i template_crop.nii.gz \
  --o template_synthseg.nii.gz --parc --cpu --threads 16 --crop 160 192 160 \
  --vol template_vol.csv --qc template_qc.csv
