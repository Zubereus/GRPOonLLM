#!/usr/bin/env bash
set -e

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

echo "Starting ablation study using OUTPUT_ROOT=${OUTPUT_ROOT}"

# 1. Main runs (1.5B)
echo "=== 1. Main Run: 1.5B with format + correctness ==="
python src/train.py \
    --run-name q15_format_correct_s42 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --reward format_correct \
    --seed 42 \
    --max-steps 300 \
    --output-root "${OUTPUT_ROOT}"

echo "=== 2. Main Run + Dense Process: 1.5B with full reward ==="
python src/train.py \
    --run-name q15_full_s42 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --reward full \
    --seed 42 \
    --max-steps 300 \
    --output-root "${OUTPUT_ROOT}"

# 2. Reward ablations (0.5B) - 2 seeds each
echo "=== 3. 0.5B Reward Ablations ==="
for reward in format correct format_correct full; do
    for seed in 42 43; do
        echo "Running 0.5B reward=${reward} seed=${seed}..."
        python src/train.py \
            --run-name "q05_${reward}_s${seed}" \
            --model Qwen/Qwen2.5-0.5B-Instruct \
            --reward "${reward}" \
            --seed "${seed}" \
            --max-steps 300 \
            --output-root "${OUTPUT_ROOT}"
    done
done

# 3. Group size ablation (G=4 on 0.5B, full reward)
echo "=== 4. Group Size Ablation (G=4) ==="
for seed in 42 43; do
    echo "Running 0.5B full reward G=4 seed=${seed}..."
    python src/train.py \
        --run-name "q05_full_g4_s${seed}" \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --reward full \
        --num-generations 4 \
        --seed "${seed}" \
        --max-steps 300 \
        --output-root "${OUTPUT_ROOT}"
done

echo "All training runs completed."
