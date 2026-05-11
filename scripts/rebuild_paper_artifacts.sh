#!/usr/bin/env bash
# Regenerate every paper artifact from results/runs/. Intended to be run after
# new runs land (either locally after `git pull` or on a workstation).
#
# Order is dependency-correct:
#   1. aggregate per-run metrics
#   2. tables (BoT-IoT, ToN-IoT, CSE-CIC stats; cross-dataset)
#   3. figures (BoT-IoT convergence, per-class; 3-dataset cross-dataset)
#   4. paper compile (pdflatex + bibtex + 2x pdflatex)
#
# Usage:
#   bash scripts/rebuild_paper_artifacts.sh
#   bash scripts/rebuild_paper_artifacts.sh --no-paper   # skip LaTeX compile

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SKIP_PAPER=0
for arg in "$@"; do
    [[ "$arg" == "--no-paper" ]] && SKIP_PAPER=1
done

echo "=== 1. aggregate ==="
python scripts/aggregate.py | tail -30

echo
echo "=== 2. cross-dataset table ==="
python scripts/build_cross_dataset_table.py

echo
echo "=== 3. BoT-IoT-specific tables ==="
python scripts/make_latex_tables.py >/dev/null

echo
echo "=== 4. per-dataset stats (Welch + paired-t + bootstrap CI) ==="
for prefix in e1_botiot e1_toniot e1_cseciic; do
    n_runs=$(ls results/runs/ 2>/dev/null | grep -c "^${prefix}_" || true)
    if [[ "$n_runs" == "0" ]]; then
        echo "  ${prefix}: skipping (no runs)"; continue
    fi
    echo "  ${prefix}: $n_runs runs"
    python scripts/stats_tests.py --prefix "$prefix" 2>&1 | tail -3
done

echo
echo "=== 5. figures ==="
python scripts/plot_convergence.py 2>&1 | tail -3
python scripts/plot_perclass.py 2>&1 | tail -3
python scripts/plot_cross_dataset.py 2>&1 | tail -3

if [[ "$SKIP_PAPER" == "1" ]]; then
    echo; echo "(skipping LaTeX compile per --no-paper)"
    exit 0
fi

echo
echo "=== 6. compile paper ==="
cd paper
pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null 2>&1 || true
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null

echo
echo "warnings/errors in final log (excluding cosmetic):"
grep -E "Warning|Overfull|Error" main.log 2>/dev/null \
    | grep -v "hyperref\|fontspec\|Underfull\|Label\|Font shape" \
    | head -10 || echo "  (none)"

ls -lh main.pdf
