"""GSM8K loading: chat prompts, gold answers, and reference step values."""
import re

from datasets import load_dataset

from rewards import parse_number

SYSTEM_PROMPT = (
    "Solve the math problem. Think step by step inside <reasoning></reasoning> "
    "tags, then put only the final number inside <answer></answer> tags.\n"
    "Format: <reasoning>...</reasoning><answer>NUMBER</answer>"
)
DEV_SIZE = 300  # last 300 train problems, held out for accuracy curves
CALC_RE = re.compile(r"<<[^>]*=([^>]*)>>")  # GSM8K calculator annotations


def to_example(row):
    solution, final = row["answer"].split("####")
    steps = [
        value
        for value in map(parse_number, CALC_RE.findall(solution))
        if value is not None
    ]
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ],
        "answer": final.strip().replace(",", ""),
        "steps": steps,
    }


def convert(dataset):
    return dataset.map(to_example, remove_columns=dataset.column_names)


def load_splits():
    raw = load_dataset("openai/gsm8k", "main")
    train_all = convert(raw["train"])
    cutoff = len(train_all) - DEV_SIZE
    return {
        "train": train_all.select(range(cutoff)),
        "dev": train_all.select(range(cutoff, len(train_all))),
        "test": convert(raw["test"]),
    }
