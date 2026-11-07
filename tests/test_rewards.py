from rewards import (
    correctness_reward,
    format_reward,
    parse_number,
    process_reward,
)

GOOD = "<reasoning>48/2 = 24. 48 + 24 = 72.</reasoning><answer>72</answer>"
WRONG = "<reasoning>48/2 = 24.</reasoning><answer>24</answer>"
UNTAGGED = "The answer is 72."
GOLD = ["72", "72", "72"]
STEPS = [[24.0, 72.0]] * 3


def test_parse_number_handles_currency_commas_and_junk():
    assert parse_number("$1,200.") == 1200.0
    assert parse_number(r"\$11") == 11.0
    assert parse_number("-5") == -5.0
    assert parse_number("about 5") is None


def test_format_reward():
    assert format_reward([GOOD, WRONG, UNTAGGED]) == [1.0, 1.0, 0.0]


def test_correctness_reward():
    rewards = correctness_reward([GOOD, WRONG, UNTAGGED], answer=GOLD)
    assert rewards == [2.0, 0.0, 0.0]


def test_process_reward_is_gated_on_correctness():
    rewards = process_reward([GOOD, WRONG, UNTAGGED], answer=GOLD, steps=STEPS)
    assert rewards == [0.75, 0.0, 0.0]


def test_chat_style_completions():
    assert format_reward([[{"role": "assistant", "content": GOOD}]]) == [1.0]
