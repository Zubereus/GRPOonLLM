# GRPO on a Small LLM: Qwen2.5-1.5B on a Free-Tier T4

A reproducible research plan and starter kit exploring:
> **Core question:** *How does reward design shape what a small LLM learns under GRPO when compute is limited?*

---

## 1. Setup & Scope

- **Model Family:** `Qwen/Qwen2.5-1.5B-Instruct` (main runs), `Qwen/Qwen2.5-0.5B-Instruct` (ablations).
- **Dataset:** GSM8K `train` split (~7.5k problems; last 300 held out for dev accuracy curves), `test` split for final evaluation.
- **Stack:** TRL `GRPOTrainer`, PEFT (LoRA), Hugging Face `transformers`, Weights & Biases.
- **Hardware Targets:**
  - **Local (GTX 1650 Ti / CPU):** Development, fast reward unit tests, data loading.
  - **Colab (free T4 16GB):** 50-step pilot and short ablations (checkpoints saved to Google Drive).
  - **Kaggle (T4 / P100 16GB):** Longer runs and multi-seed ablations.

---

## 2. Directory Structure

```
├── README.md                 # Core question, results table, reproduction guide
├── requirements.txt          # Python dependencies
├── pytest.ini                # Puts src/ on python path
├── .gitignore                # outputs/, wandb/, caches
├── configs/
│   └── base.yaml             # T4-sized GRPO + LoRA hyperparameters
├── src/
│   ├── data.py               # GSM8K loader -> chat prompt, gold answer, steps
│   ├── rewards.py            # format / correctness / dense process rewards
│   ├── train.py              # GRPOTrainer entry point with auto-resume
│   └── evaluate.py           # Greedy eval, accuracy, format rate, completions dump
├── tests/
│   └── test_rewards.py       # Reward unit tests (CPU, passes locally)
├── scripts/
│   ├── run_baseline.sh       # Untouched model evaluations
│   └── run_ablations.sh      # Fixed run matrix script
├── notebooks/
│   ├── colab_run.ipynb       # Colab notebook (Drive mount, pilot, training)
│   └── kaggle_run.ipynb      # Kaggle notebook
├── results/                  # Dev/test summaries and completion samples
├── report/
│   ├── analysis.md           # Failure cases, reward hacking observations
│   └── figures/              # Training dynamics & ablation plots
└── outputs/                  # Model checkpoints (gitignored)
```

---

## 3. Reward Formulations

Four reward configurations can be selected via `--reward <variant>`:

| Variant | Functions | Max Reward | Expected Behavior |
|---|---|---|---|
| `format` | format | 1.0 | Fast tag compliance (`<reasoning>...</reasoning><answer>...</answer>`); test of reward-hacking without reasoning accuracy improvement. |
| `correct` | correctness | 2.0 | Sparse signal; slower start if format or accuracy is initially low. |
| `format_correct` | format + correctness | 3.0 | Standard combined baseline from project plan. |
| `full` | format + correctness + process | 3.75 | Tests whether intermediate calculation step credit (adapted from H-GRPO) stabilizes small models. |

---

## 4. Runbook & Reproduction

### Step 1: Run Unit Tests Locally
```bash
pip install pytest pyyaml datasets
pytest
```
*All 5 unit tests in `tests/test_rewards.py` must pass.*

### Step 2: Untouched Baselines (Dev Set)
```bash
python src/evaluate.py --base Qwen/Qwen2.5-0.5B-Instruct --split dev --out results/base_0.5b
python src/evaluate.py --base Qwen/Qwen2.5-1.5B-Instruct --split dev --out results/base_1.5b
```

### Step 3: 50-Step Pilot Run (Colab / T4)
Benchmark seconds per step and peak VRAM:
```bash
python src/train.py --run-name pilot_1.5b --max-steps 50
```

### Step 4: Main Training Run
```bash
python src/train.py \
    --run-name q15_format_correct_s42 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --reward format_correct \
    --seed 42 \
    --max-steps 300 \
    --output-root outputs
```

### Step 5: Full Run Matrix / Ablations
```bash
bash scripts/run_ablations.sh
```

---

## 5. Hyperparameter Highlights (`configs/base.yaml`)

- **Precision:** `fp16: true`, `bf16: false` (optimized for T4 architecture).
- **Group Size ($G$):** 8 rollouts per prompt (4 in group size ablation).
- **Batching:** Micro-batch size 4 $\times$ accumulation 8 = 32 completions per step (4 prompts $\times G=8$).
- **KL Beta:** `0.0` (skips reference-model pass to conserve VRAM).
- **LoRA:** $r=16, \alpha=32$, targeting `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]`.
