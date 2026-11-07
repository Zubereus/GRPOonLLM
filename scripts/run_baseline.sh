#!/usr/bin/env bash
set -e

# Baselines on the dev split
echo "Evaluating untouched Qwen2.5-0.5B-Instruct baseline..."
python src/evaluate.py --base Qwen/Qwen2.5-0.5B-Instruct --split dev --out results/base_0.5b

echo "Evaluating untouched Qwen2.5-1.5B-Instruct baseline..."
python src/evaluate.py --base Qwen/Qwen2.5-1.5B-Instruct --split dev --out results/base_1.5b
