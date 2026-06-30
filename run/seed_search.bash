#!/bin/bash
# seed_search.bash — Run multiple seeds for a short trial, pick the best by val_unseen SR
# Usage: bash run/seed_search.bash <gpu_id>
# Example: bash run/seed_search.bash 0

set -e

GPU=${1:?Usage: bash run/seed_search.bash <gpu_id>}
SEEDS=(1 42 123 456 789)
ITERS=3000
RESULTS_FILE="/tmp/seed_search_results.txt"

# Common training flags (same as agent.bash, minus iters/name/seed)
BASE_FLAGS="--attn soft --train listener
      --features rn50x4
      --feature_size 640
      --batchSize 64
      --featdropout 0.3
      --angleFeatSize 128
      --feedback sample
      --mlWeight 0.2
      --option_size 8
      --option_step 3
      --entropyCoef 0.025
      --criticLr 2e-4
      --subout max --dropout 0.2 --optim adam --lr 1e-4 --maxAction 15"

> "$RESULTS_FILE"

echo "========================================"
echo "  DILLM Seed Search"
echo "  Seeds: ${SEEDS[*]}"
echo "  Iterations per seed: $ITERS"
echo "  Metric: val_unseen success_rate"
echo "  GPU: $GPU"
echo "========================================"

for seed in "${SEEDS[@]}"; do
    name="seed_search_${seed}"
    logfile="/tmp/seed_search_${seed}.log"
    echo ""
    echo ">>> [$(date '+%H:%M:%S')] Running seed=$seed (name=$name) ..."
    mkdir -p "snap/$name"

    # Run training, save output to log file
    CUDA_VISIBLE_DEVICES=$GPU CUDA_LAUNCH_BLOCKING=1 \
        python r2r_src/train.py $BASE_FLAGS \
        --seed "$seed" --iters "$ITERS" --name "$name" \
        2>&1 | tee "$logfile"

    # Parse best val_unseen SR from BEST RESULT block
    # Line format: "val_unseen Iter NNN , val_unseen , ..., success_rate: 0.XXX, ..."
    best_sr=$(grep "^val_unseen Iter" "$logfile" \
        | tail -1 \
        | grep -oP 'success_rate: \K[0-9.]+' \
        | head -1)

    # Fallback: if BEST RESULT never printed, parse last regular validation line
    if [ -z "$best_sr" ]; then
        best_sr=$(grep "val_unseen.*success_rate" "$logfile" \
            | tail -1 \
            | grep -oP 'success_rate: \K[0-9.]+' \
            | head -1)
    fi

    if [ -z "$best_sr" ]; then
        best_sr="0.000"
        echo ">>> WARNING: Could not parse val_unseen SR for seed $seed"
    fi

    echo ">>> [$(date '+%H:%M:%S')] Seed $seed done — best val_unseen SR = $best_sr"
    echo "$seed $best_sr" >> "$RESULTS_FILE"
done

echo ""
echo "========================================"
echo "  Seed Search Results"
echo "========================================"
echo ""
printf "  %-10s %-20s\n" "Seed" "val_unseen SR"
printf "  %-10s %-20s\n" "----" "-------------"
while read -r s sr; do
    printf "  %-10s %-20s\n" "$s" "$sr"
done < "$RESULTS_FILE"

# Find best seed (sort by SR descending, take first)
best_line=$(sort -k2 -rn "$RESULTS_FILE" | head -1)
best_seed=$(echo "$best_line" | awk '{print $1}')
best_sr=$(echo "$best_line" | awk '{print $2}')

echo ""
echo "  >>> Best seed: $best_seed (val_unseen SR = $best_sr)"
echo ""

# Write best seed into run/agent.bash
if grep -q -- '--seed' run/agent.bash; then
    sed -i "s/--seed [0-9]*/--seed $best_seed/" run/agent.bash
    echo ">>> Updated run/agent.bash: --seed $best_seed"
else
    sed -i "s/--subout/--seed $best_seed\n      --subout/" run/agent.bash
    echo ">>> Inserted --seed $best_seed into run/agent.bash"
fi

echo ">>> To start full training: bash run/agent.bash <gpu_id>"
echo ">>> Seed search logs saved in /tmp/seed_search_*.log"
