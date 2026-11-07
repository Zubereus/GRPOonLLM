"""GRPO training on GSM8K. Run from the repo root: python src/train.py ..."""
import argparse
import os

import torch
import yaml
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

from data import load_splits
from rewards import REWARD_SETS


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO on GSM8K")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--reward", choices=sorted(REWARD_SETS), default="full")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-root", default="outputs")
    return parser.parse_args()


def load_settings(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def build_grpo_config(args, settings):
    return GRPOConfig(
        output_dir=os.path.join(args.output_root, args.run_name),
        run_name=args.run_name,
        seed=args.seed,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        **settings["grpo"],
    )


def latest_checkpoint(output_dir):
    return get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None


def main():
    args = parse_args()
    settings = load_settings(args.config)
    trainer = GRPOTrainer(
        # `dtype` needs transformers >= 4.56; older versions call it torch_dtype
        model=AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16),
        processing_class=AutoTokenizer.from_pretrained(args.model),
        reward_funcs=REWARD_SETS[args.reward],
        args=build_grpo_config(args, settings),
        train_dataset=load_splits()["train"].shuffle(seed=args.seed),
        peft_config=LoraConfig(task_type="CAUSAL_LM", **settings["lora"]),
    )
    output_dir = trainer.args.output_dir
    trainer.train(resume_from_checkpoint=latest_checkpoint(output_dir))
    trainer.save_model(os.path.join(output_dir, "final"))


if __name__ == "__main__":
    main()
