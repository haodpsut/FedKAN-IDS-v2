#!/usr/bin/env bash
# Server-side equivalent of notebook cell 4d. Auto-commits + retries pushes,
# survives disconnects via skip-existing.
#
# Usage:
#   bash scripts/run_grid.sh                    # full M2 grid on BoT-IoT (72 runs)
#   DATASET=toniot bash scripts/run_grid.sh     # M3c on ToN-IoT (72 runs)
#   MINIMAL=1 DATASET=toniot bash scripts/run_grid.sh   # only Dir(0.1) cell (24 runs)
#   FILL_M3A=1 bash scripts/run_grid.sh          # add the 7 extra seeds for n=10 stats

set -euo pipefail

DATASET="${DATASET:-botiot}"        # botiot | toniot | cseciic
MINIMAL="${MINIMAL:-0}"
FILL_M3A="${FILL_M3A:-0}"
PUSH_EVERY_N="${PUSH_EVERY_N:-4}"

if [[ "$DATASET" == "botiot" ]]; then
    DATA_NAME="nf_botiot_v2"; EXP_PREFIX="e1_botiot"
elif [[ "$DATASET" == "toniot" ]]; then
    DATA_NAME="nf_toniot_v2"; EXP_PREFIX="e1_toniot"
elif [[ "$DATASET" == "cseciic" ]]; then
    DATA_NAME="nf_cseciic_v2"; EXP_PREFIX="e1_cseciic"
else
    echo "Unknown DATASET=$DATASET (use botiot|toniot|cseciic)"; exit 1
fi

# Patch the base config to point at the requested dataset
RUNTIME_CFG="configs/experiments/_${EXP_PREFIX}_runtime.yaml"
python -c "
import yaml
with open('configs/experiments/e1_botiot.yaml') as f: c = yaml.safe_load(f)
c['data']['name'] = '$DATA_NAME'
c['experiment']['id'] = '$EXP_PREFIX'
with open('$RUNTIME_CFG', 'w') as f: yaml.safe_dump(c, f, sort_keys=False)
print('wrote', '$RUNTIME_CFG')
"

VARIANTS=(
    "kan_h8 kan 8 5"
    "mlp_h32 mlp 32 -"
    "mlp_h80 mlp 80 -"
    "kan_h16 kan 16 5"
)

if [[ "$MINIMAL" == "1" ]]; then
    PARTITIONS=("dirichlet 0.1")
else
    PARTITIONS=("iid -" "dirichlet 1.0" "dirichlet 0.1")
fi
MODES=("binary 130000" "multiclass 50000")

if [[ "$FILL_M3A" == "1" ]]; then
    SEEDS=(11 17 23 29 31 37 43)
    PARTITIONS=("dirichlet 0.1")        # M3a only adds seeds to the Dir(0.1) cell
else
    SEEDS=(42 2024 2026)
fi

# Build the plan as a flat list to pass through bash safely
PLAN=()
for mp in "${MODES[@]}"; do
    read -r mode ds <<<"$mp"
    for v in "${VARIANTS[@]}"; do
        read -r tag mname hidden grid <<<"$v"
        for p in "${PARTITIONS[@]}"; do
            read -r ptype alpha <<<"$p"
            for seed in "${SEEDS[@]}"; do
                PLAN+=("$mode|$ds|$tag|$mname|$hidden|$grid|$ptype|$alpha|$seed")
            done
        done
    done
done

TOTAL=${#PLAN[@]}
echo "Grid plan: $TOTAL runs (DATASET=$DATASET MINIMAL=$MINIMAL FILL_M3A=$FILL_M3A)"

push_chunk() {
    local label="$1"
    git add results/
    if git commit -m "results: ${EXP_PREFIX} partial ${label} $(date -u +%Y-%m-%dT%H:%M:%SZ)" >/dev/null 2>&1; then
        for attempt in 1 2 3; do
            if git push origin main >/dev/null 2>&1; then
                echo "  [push] $label OK (attempt $attempt)"; return
            fi
            echo "  [push] $label attempt $attempt failed; retry"; sleep $((attempt * 2))
        done
        echo "  [push] $label ALL RETRIES FAILED — keep results local"
    else
        echo "  [push] $label nothing to commit"
    fi
}

i=0
for spec in "${PLAN[@]}"; do
    i=$((i+1))
    IFS='|' read -r mode ds tag mname hidden grid ptype alpha seed <<<"$spec"
    exp_id="${EXP_PREFIX}_${mode}_${tag}"
    extra=()
    [[ "$alpha" != "-" ]] && extra+=(--alpha "$alpha")
    [[ "$grid"  != "-" ]] && extra+=(--grid-size "$grid")
    t0=$(date +%s)
    set +e
    out=$(python scripts/run_experiment.py \
        --config "$RUNTIME_CFG" \
        --seed "$seed" --skip-existing \
        --exp-id "$exp_id" \
        --model-name "$mname" --hidden "$hidden" \
        --mode "$mode" --downsample "$ds" \
        --partition "$ptype" "${extra[@]}" 2>&1)
    rc=$?
    set -e
    dt=$(( $(date +%s) - t0 ))
    summary=$(echo "$out" | grep -E 'DONE|SKIP|Error|Traceback' | head -1 || echo "exit=$rc")
    echo "[$i/$TOTAL] ${exp_id}/${ptype}${alpha}/seed${seed}  t=${dt}s  ${summary:0:120}"
    if (( i % PUSH_EVERY_N == 0 || i == TOTAL )); then
        push_chunk "through $i/$TOTAL"
    fi
done

echo "DONE — $TOTAL runs processed."
