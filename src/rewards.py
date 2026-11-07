"""Reward functions for GRPO on GSM8K.

TRL calls each function with `completions` plus every extra dataset column
(here `answer` and `steps`) as keyword arguments, and expects one float per
completion.
"""
import re
from collections import Counter

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)
FORMAT_RE = re.compile(
    r"^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$", re.DOTALL
)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

GAMMA = 0.25  # step coverage needed before the process reward switches on


def completion_text(completion):
    """Chat-style completions are [{"role": ..., "content": ...}]."""
    return completion[0]["content"] if type(completion) is list else completion


def parse_number(text):
    cleaned = re.sub(r"[,$%\s\\]", "", text).rstrip(".")
    match = NUMBER_RE.fullmatch(cleaned)
    return float(match.group()) if match else None


def extract_answer(text):
    matches = ANSWER_RE.findall(text)
    return parse_number(matches[-1]) if matches else None


def is_correct(text, gold):
    predicted = extract_answer(text)
    return predicted is not None and abs(predicted - float(gold)) < 1e-6


def reasoning_of(text):
    match = REASONING_RE.search(text)
    return match.group(1) if match else text


def number_counts(text):
    tokens = NUMBER_RE.findall(text.replace(",", ""))
    return Counter(round(float(token), 4) for token in tokens)


def step_coverage(text, steps):
    """Share of reference intermediate results found in the reasoning.

    One-to-one and order-invariant (multiset matching), in the spirit of the
    H-GRPO reward but without needing the Hungarian algorithm: the matches
    here are exact number equalities.
    """
    produced = number_counts(text)
    wanted = Counter(round(float(step), 4) for step in steps)
    matched = sum(min(count, produced[value]) for value, count in wanted.items())
    return matched / len(steps) if steps else 0.0


def format_reward(completions, **kwargs):
    """1.0 if the completion matches <reasoning>..</reasoning><answer>..</answer>"""
    return [
        1.0 if FORMAT_RE.match(completion_text(c)) else 0.0 for c in completions
    ]


def correctness_reward(completions, answer, **kwargs):
    """2.0 if the number inside <answer> equals the gold answer."""
    return [
        2.0 if is_correct(completion_text(c), gold) else 0.0
        for c, gold in zip(completions, answer)
    ]


def process_reward(completions, answer, steps, **kwargs):
    """Dense bonus (max 1 - GAMMA), gated on a correct final answer."""
    texts = [completion_text(c) for c in completions]
    return [
        (1.0 if is_correct(text, gold) else 0.0)
        * max(0.0, step_coverage(reasoning_of(text), reference) - GAMMA)
        for text, gold, reference in zip(texts, answer, steps)
    ]


REWARD_SETS = {
    "format": [format_reward],
    "correct": [correctness_reward],
    "format_correct": [format_reward, correctness_reward],
    "full": [format_reward, correctness_reward, process_reward],
}
