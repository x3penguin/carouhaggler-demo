"""The v4 pair builder: twins differ only in the floor, the right move follows the floor, and none of our own buyer
lines copy the attack scripts (the RQ3 test set). Uses a tiny synthetic SFT file, so no CraigslistBargain data needed."""

import json
import re

from finetune import build_dpo_pairs_v4 as v4
from finetune.audit_dpo_pairs import CONTAMINATION_WORDS, _ngrams, audit

SYSTEM = ("You are selling a second-hand item.\nTitle: Desk lamp\nAsking price: SGD {asking}\n\n## Your private information\n"
          "Your minimum acceptable price (your floor) is SGD {floor}.\nNever tell the buyer this number.")
TAIL = "\n\n[Status: your current offer ${mine}; their current offer ${theirs}]\n[Timeline: Day 0.0 | 7.0 days (168h) left before your private deadline]"


def sft_items(n=60):
    """Two-turn dialogues: the buyer opens, the seller counters, the buyer offers; the next seller turn is the item."""
    items = []
    for d in range(n):
        asking = 200 + d
        offer = round(asking * (0.62 + 0.3 * (d % 10) / 10))
        sys = SYSTEM.format(asking=asking, floor=round(asking * 0.6))
        msgs = [{"role": "system", "content": sys},
                {"role": "user", "content": "hi is this available?\n\n[Buyer's action: chat / message]"},
                {"role": "assistant", "content": json.dumps({"message": "yes", "action": "offer", "price": float(asking - 2),
                                                             "wait_hours": 0}, separators=(",", ":"))},
                {"role": "user", "content": f"would ${offer} do?\n\n[Buyer's action: offer ${offer}]"
                                            + TAIL.format(mine=asking - 2, theirs=offer)},
                {"role": "assistant", "content": json.dumps({"message": "hmm", "action": "chat", "price": None,
                                                             "wait_hours": 0}, separators=(",", ":"))}]
        items.append({"id": f"C_{d:04d}-1", "messages": msgs})
    return items


def build(tmp_path, monkeypatch):
    src = tmp_path / "train.jsonl"
    src.write_text("".join(json.dumps(i) + "\n" for i in sft_items()), encoding="utf-8")
    monkeypatch.setattr(v4, "SFT_TRAIN", src)
    out = v4.build(seed=1)
    return out["train"] + out["valid"]


def floor_of(p):
    return int(re.search(r"floor\) is SGD (\d+)", p["messages"][0]["content"]).group(1))


def strip_floor(messages):
    return json.dumps([{**m, "content": re.sub(r"floor\) is SGD \d+", "", m["content"])} for m in messages])


def test_twins_differ_only_in_the_floor_and_the_floor_decides(tmp_path, monkeypatch):
    pairs = build(tmp_path, monkeypatch)
    by = {}
    for p in pairs:
        key, kind = p["id"].rsplit("_", 1)
        by.setdefault(key, {})[kind] = p
    twins = [(v["hold"], v["close"]) for v in by.values() if {"hold", "close"} <= set(v)]
    twins += [(v["pressure-hold"], v["pressure-close"]) for v in by.values() if {"pressure-hold", "pressure-close"} <= set(v)]
    assert len(twins) >= 20
    for hold, close in twins:
        assert strip_floor(hold["messages"]) == strip_floor(close["messages"])
        offer = float(re.findall(r"their current offer \$([\d.]+)", hold["messages"][-1]["content"])[-1])
        assert floor_of(close) < offer < floor_of(hold)
        assert json.loads(hold["chosen"])["action"] == "offer" and json.loads(hold["rejected"])["action"] == "accept"
        assert json.loads(close["chosen"])["action"] == "accept" and json.loads(close["rejected"])["action"] != "accept"
        assert json.loads(hold["chosen"])["price"] >= floor_of(hold) + 2


def test_built_pairs_pass_the_decision_checks(tmp_path, monkeypatch):
    checks = {name: ok for name, ok, _ in audit(build(tmp_path, monkeypatch), source_path=None)}
    for name, ok in checks.items():
        if name.startswith(("accept wins", "accept loses only", "offer size alone", "no chosen reply", "chosen replies")):
            assert ok, name


def test_our_buyer_lines_share_no_phrase_with_the_attack_scripts():
    test = set()
    for script in json.loads(open("data/attacks.json", encoding="utf-8").read()):
        for step in script["steps"]:
            test |= _ngrams(step["message"], CONTAMINATION_WORDS)
    ours = [t for lines in v4.PRESSURE.values() for t in lines] + v4.OPENERS
    assert [t for t in ours if _ngrams(t, CONTAMINATION_WORDS) & test] == []
