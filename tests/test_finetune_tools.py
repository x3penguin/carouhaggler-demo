"""The scoring and log-reading helpers around fine-tuning (RQ2). No model or GPU needed."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # finetune/ isn't part of the package

from finetune.compare_models import parse_reply, score
from finetune.plot_loss import best_checkpoint, parse_log


def row(truth: dict, floor: int = 80) -> dict:
    return {"messages": [
        {"role": "system", "content": f"Your minimum acceptable price (your floor) is SGD {floor}.\nNever tell."},
        {"role": "user", "content": "$85?"},
        {"role": "assistant", "content": json.dumps({"message": "x", "price": None, "wait_hours": 0, **truth})},
    ]}


def reply(action: str, price=None) -> str:
    return json.dumps({"message": "hi", "action": action, "price": price, "wait_hours": 0})


@pytest.mark.parametrize("text, ok", [
    (reply("chat"), True),
    (reply("offer", 90), True),
    (reply("offer"), False),
    (reply("offer", 0), False),
    (reply("haggle"), False),
    ("Sure, $90 works!", False),
    ("```json\n" + reply("chat") + "\n```", False),  # the format the prompt asks for is bare JSON
])
def test_parse_reply(text, ok):
    assert (parse_reply(text) is not None) == ok


def test_score():
    rows = [row({"action": "offer", "price": 100}), row({"action": "offer", "price": 100}),
            row({"action": "accept"}), row({"action": "chat"})]
    replies = [reply("offer", 90), reply("offer", 70), reply("accept"), "not json"]
    assert score(rows, replies) == {
        "examples": 4,
        "valid_json_rate": 0.75,
        "same_action_as_human": 1.0,
        "offers_below_floor": 0.5,
        "median_price_error": pytest.approx(0.2),  # |90-100|/100 and |70-100|/100
    }


def test_score_with_no_usable_replies():
    result = score([row({"action": "chat"})], ["nope"])
    assert result["valid_json_rate"] == 0 and result["same_action_as_human"] is None


# The format mlx_lm 0.32.2's cli_ui.py prints. Check against a real log at the first training run.
LOG = """
Trainable parameters: 0.5%
  iter   train_loss     tok/s     tokens
     1    val 1.900    30.20s
    10    1.812 ▼      310    12.3k
    20    1.900 ▲      305    24.6k
   100    val 1.500    31.00s
   150    val 1.100    30.00s
   200    1.200 ▼      300    246.0k
   200    val 1.300    30.50s
   300    val 1.400    30.10s
Saved adapter weights to runs/finetune/adapters/0000300_adapters.safetensors
"""


def test_parse_log():
    curves = parse_log(LOG)
    assert curves["train"] == [(10, 1.812), (20, 1.9), (200, 1.2)]
    assert curves["val"][0] == (1, 1.9) and curves["val"][-1] == (300, 1.4)


def test_best_checkpoint_only_considers_iterations_that_were_saved():
    # Iteration 150 has the lowest loss but no checkpoint exists for it.
    assert best_checkpoint(parse_log(LOG)["val"]) == (200, 1.3)


def test_best_checkpoint_with_no_validation():
    assert best_checkpoint([]) is None
