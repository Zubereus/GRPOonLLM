# Analysis & Findings

## 1. Research Question
*How does reward design shape what a small LLM learns under GRPO when compute is limited?*

## 2. Evaluation Results Matrix

| Block | Model | Reward Variant | G | Seed | Dev Acc (%) | Lenient Acc (%) | Acc \| format (%) | Test Acc (%) | Format Rate (%) | Avg Length (chars) | Avg Length (tokens) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | Qwen2.5-0.5B-Instruct | Untouched | - | - | 25.0 | 52.7 | 50.6 | - | 26.3 | 569.8 | 185.1 |
| Baseline | Qwen2.5-1.5B-Instruct | Untouched | - | - | 62.3 | 67.0 | 68.2 | - | 86.0 | 688.8 | 211.3 |
| Main | Qwen2.5-1.5B-Instruct | format_correct | 8 | 42 | | | | | | | |
| Main + Dense | Qwen2.5-1.5B-Instruct | full | 8 | 42 | | | | | | | |
| Ablation: Reward | Qwen2.5-0.5B-Instruct | format | 8 | 42 | | | | | | | |
| Ablation: Reward | Qwen2.5-0.5B-Instruct | format | 8 | 43 | | | | | | | |
| Ablation: Reward | Qwen2.5-0.5B-Instruct | correct | 8 | 42 | | | | | | | |
| Ablation: Reward | Qwen2.5-0.5B-Instruct | correct | 8 | 43 | | | | | | | |
| Ablation: Reward | Qwen2.5-0.5B-Instruct | format_correct | 8 | 42 | | | | | | | |
| Ablation: Reward | Qwen2.5-0.5B-Instruct | format_correct | 8 | 43 | | | | | | | |
| Ablation: Reward | Qwen2.5-0.5B-Instruct | full | 8 | 42 | | | | | | | |
| Ablation: Reward | Qwen2.5-0.5B-Instruct | full | 8 | 43 | | | | | | | |
| Ablation: Group Size | Qwen2.5-0.5B-Instruct | full (G=4) | 4 | 42 | | | | | | | |
| Ablation: Group Size | Qwen2.5-0.5B-Instruct | full (G=4) | 4 | 43 | | | | | | | |

## 3. Baseline Observations & Ceilings

### 3.1 Format-Only vs. Reasoning Ceilings
- **0.5B Baseline**: Strict accuracy is **25.0%** (75/300), but lenient accuracy (last number in completion equals gold) reaches **52.7%** (158/300). Exactly 83 completions calculate the correct answer but miss strict credit due to absent `<answer>` tags. When the model does follow format, accuracy is **50.6%** (40/79).
- **1.5B Baseline**: Strict accuracy is **62.3%** (187/300, accounting for escaped currency `\$`), with lenient accuracy at **67.0%** (201/300).
- **Implication**: The approximate format-only ceilings are ~53% (0.5B) and ~67% (1.5B). Accuracy gains up to these levels reflect tag/format acquisition. Only gains above these ceilings demonstrate improved mathematical reasoning. Both metrics will be tracked across all runs.

### 3.2 Prompt & SFT Decision (Step 3)
- The 0.5B model achieves 25.0% strict reward in the zero-shot untouched state, meaning roughly 1 in 4 rollouts receives positive reward signal immediately.
- We skip few-shot prompting and SFT warm-up: the initial policy is not reward-starved, and observing natural format acquisition under RL is central to the research question.

### 3.3 Dev Set Standard Error & Final Evaluations
- With $n=300$, the standard error on the dev split is $\approx 2.5\text{--}2.8\%$, yielding a $95\%$ confidence interval of approximately $\pm 5\text{--}6\%$.
- Fine-grained differences (such as dense process rewards vs. sparse rewards) will be evaluated on the full 1,319 GSM8K test set using paired seeds (42 and 43).

### 3.4 Completion Lengths & Budgeting
- Mean completion token lengths are 185.1 tokens (0.5B) and 211.3 tokens (1.5B).
- The `max_completion_length: 320` budget accommodates the vast majority of solutions, but `completions/clipped_ratio` will be monitored during the 50-step pilot run.

## 4. Failure Cases and Reward Hacking Observations

### 4.1 Format-only Reward Hacking
*Document instances where model satisfies `<reasoning>...</reasoning><answer>...</answer>` perfectly while outputting incorrect or nonsensical reasoning.*

### 4.2 Length Drift
*Observations on completion length during training under sparse vs. dense rewards.*

### 4.3 Reasoning Quality & Process Reward Impact
*Impact of intermediate calculator steps bonus on accuracy and coherence.*
