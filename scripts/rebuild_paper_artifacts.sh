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

# --- Vong R1 (08/2026). Bang cu o tren sinh tu results/runs (RTX 4090); nhung
# cai duoi day sinh tu run tren MOT may CPU. Truoc 20/08 script nay khong goi
# cai nao trong so chung, nen cau "regenerates every table" trong bai la SAI.
# lr_sweep.py KHONG goi o day: no CHAY thi nghiem, viec do thuoc run_all_local.sh.
echo "[R1] bang headline tu run mot may"
python scripts/make_tables_r1.py 2>&1 | tail -3
echo "[R1] phan tich sweep learning rate (4 o)"
for c in botiot toniot cseciic cseciic_mc50k; do
  python scripts/analyse_sweep.py results/lrsweep_$c 2>&1 | tail -4
done
echo "[R1] giao thuc long nhau: chon lr tren seed giu rieng"
python scripts/lr_holdout.py 2>&1 | tail -6
echo "[R1] chi phi suy dien tren CPU mot luong"
python scripts/hw_profile.py --reps 300 --out results/hw_profile_idle.json 2>&1 | tail -4
echo "[R1] metric IDS suy tu run da co (MCC, FPR)"
python scripts/derive_ids_metrics.py 2>&1 | tail -4
echo "[R1] chi phi truyen: byte toi dich"
python scripts/comm_cost.py 2>&1 | tail -4
echo "[R1] phap y phan hoach seed 17"
python scripts/seed17_forensics.py --config configs/experiments/e1_toniot.yaml --mode binary 2>&1 | tail -4
echo "[R1] pho Hessian tai nghiem (R2#6)"
python3 scripts/hessian_spectrum.py || true
echo "[R1] do nhay theo G / bac spline / do rong (R2#5)"
python3 scripts/sensitivity_grid.py || true
echo "[R1] TRUY SO cho 7 bang GO TAY trong bai"
python3 scripts/verify_handtyped_tables.py || true
echo "[R1] bang nguon cho cong truy so"
python scripts/emit_claims_csv.py 2>&1 | tail -2

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
