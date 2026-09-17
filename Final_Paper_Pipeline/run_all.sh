#!/usr/bin/env bash
# Regenerate every figure and every reported statistic, front to back, on one command.
#
# Usage: ./run_all.sh
# Requires the `final_paper_pipeline` conda env (see environment.yml) to be active,
# or edit PYTHON below to point at an interpreter with those packages installed.
#
# The Jupyter kernel is pinned explicitly (KERNEL_NAME below, registered once via
# `python -m ipykernel install --user --name final_paper_pipeline ...`) rather than
# left to nbconvert's default kernel resolution: this machine has several
# identically-named "python3" kernelspecs registered across different environments
# (conda base, a system Python, etc.), and which one nbconvert picks by default is
# not guaranteed -- it silently ran one Fig1 notebook execution on the wrong
# interpreter/matplotlib version during development, producing a spurious
# TypeError. Pinning the kernel name removes that ambiguity.

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
KERNEL_NAME="${KERNEL_NAME:-final_paper_pipeline}"

echo "=== Final_Paper_Pipeline: run_all ==="
echo "Stages currently implemented: Fig1 (all 4 sub-stages), Fig2, Fig3, Fig4, Fig5 (panels a-d),"
echo "  Fig6, Fig7 (panels a-h, canonical 7-state taxonomy), SuppFig1 (all 5 panels; a-b are"
echo "  placeholders, not data-derived), SuppFig2, SuppFig3 (all 7 panels), SuppFig4,"
echo "  Fig8 (rebuilt: panels a-f on the canonical 7-state taxonomy)"
echo

for nb in notebooks/Fig1.ipynb notebooks/Fig2.ipynb notebooks/Fig3.ipynb notebooks/Fig4.ipynb notebooks/Fig5.ipynb notebooks/Fig6.ipynb notebooks/Fig7.ipynb notebooks/SuppFig1.ipynb notebooks/SuppFig2.ipynb notebooks/SuppFig3.ipynb notebooks/SuppFig4.ipynb notebooks/Fig8.ipynb; do
  echo "--- Executing $nb ---"
  "$PYTHON" -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name="$KERNEL_NAME" "$nb"
done

echo
echo "=== Done. Figures in outputs/figures/, values manifests in outputs/values_manifests/ ==="
