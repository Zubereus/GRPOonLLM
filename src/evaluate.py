"""Greedy evaluation: accuracy and format rate on GSM8K dev or test."""
import argparse
import json
import os
from statistics import mean

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_splits
from rewards import FORMAT_RE, NUMBER_RE, is_correct


def last_number(text):
    tokens = NUMBER_RE.findall(text.replace(",", ""))
    return float(tokens[-1]) if tokens else None


def lenient_correct(text, gold):
    value = last_number(text)
    return value is not None and abs(value - float(gold)) < 1e-6


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on GSM8K")
    parser.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default=None, help="LoRA checkpoint dir")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_model(base, adapter):
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.float16, device_map="auto"
    )
    return PeftModel.from_pretrained(model, adapter) if adapter else model


def batches(rows, size):
    return [rows[start:start + size] for start in range(0, len(rows), size)]


def generate_batch(model, tokenizer, prompts, max_new_tokens):
    texts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    return tokenizer.batch_decode(
        output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base, padding_side="left")
    model = load_model(args.base, args.adapter).eval()
    rows = list(load_splits()[args.split])[: args.limit]
    completions = [
        text
        for batch in batches(rows, args.batch_size)
        for text in generate_batch(
            model, tokenizer, [row["prompt"] for row in batch], args.max_new_tokens
        )
    ]
    hits = [is_correct(c, r["answer"]) for c, r in zip(completions, rows)]
    lenient_hits = [lenient_correct(c, r["answer"]) for c, r in zip(completions, rows)]
    formatted_flags = [bool(FORMAT_RE.match(c)) for c in completions]
    formatted_hits = [h for h, f in zip(hits, formatted_flags) if f]
    token_lengths = [len(tokenizer(c).input_ids) for c in completions]

    summary = {
        "n": len(rows),
        "accuracy": mean(hits),
        "lenient_accuracy": mean(lenient_hits),
        "format_rate": mean(formatted_flags),
        "acc_given_format": mean(formatted_hits) if formatted_hits else 0.0,
        "mean_chars": mean(len(c) for c in completions),
        "mean_tokens": mean(token_lengths),
    }
    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, f"{args.split}_summary.json")
    samples_path = os.path.join(args.out, f"{args.split}_completions.jsonl")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(samples_path, "w") as handle:
        handle.write(
            "\n".join(
                json.dumps({"gold": r["answer"], "completion": c})
                for c, r in zip(completions, rows)
            )
        )
    print(summary)


if __name__ == "__main__":
    main()
