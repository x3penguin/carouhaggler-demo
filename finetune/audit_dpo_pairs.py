"""Check a DPO preference file before training on it (RQ2 v3; see finetune/DPO_V3.md).

Run from the repo root:  uv run python -m finetune.audit_dpo_pairs path/to/train.jsonl

DPO learns from the *difference* between the chosen and the rejected reply, so a pattern that
only ever sits on one side gets learned whether or not we meant it. v2 had `accept` as the
rejected move in 559 pairs and never as the chosen move against a non-accept, so the model
learned "accepting is bad" and nearly stopped closing deals. These checks catch that kind of
imbalance, plus the other flaws found in v2, before a training run is spent on it.

Each line is {"id", "messages", "chosen", "rejected"}: `messages` is the chat prompt (system
prompt first, buyer's last turn last) and chosen/rejected are the seller's JSON replies. The
pair type is the id's suffix after the last "_" (e.g. `..._subfloor`). Exit code 1 if a check fails.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

FLOOR = re.compile(r"your floor\) is SGD ([\d,.]+)")
BUYER_OFFER = re.compile(r"their current offer \$([\d,.]+)")
# Words that tell the buyer the deal is done. If the action isn't `accept`, the chat and the move disagree.
SAYS_DEAL = re.compile(r"^\W*(ok(ay)?[, ]+)?(deal|it'?s a deal|sold|done|agreed)\b", re.IGNORECASE)

MAX_SHARE_ONE_TEXT = 0.05   # no single reply text may be more than 5% of one side of a pair type
MAX_SHARE_ONE_TYPE = 0.40   # no pair type may be more than 40% of the file


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def parse(pair: dict) -> dict:
    """The facts each check needs: floor, the buyer's standing offer, and both replies."""
    system, last = pair["messages"][0]["content"], pair["messages"][-1]["content"]
    floor, offer = FLOOR.search(system), BUYER_OFFER.findall(last)
    return {"type": pair["id"].rsplit("_", 1)[-1],
            "floor": _num(floor.group(1)) if floor else None,
            "buyer_offer": _num(offer[-1]) if offer else None,
            "chosen": json.loads(pair["chosen"]), "rejected": json.loads(pair["rejected"])}


def _states_floor(r: dict) -> bool:
    """A number within $1 of the floor in the chosen message, the same rule as the floor-leak metric. Repeating the
    buyer's own offer back ("I can't go as low as $85") doesn't count, even when that offer happens to sit near the floor."""
    if r["floor"] is None:
        return False
    for x in re.findall(r"\d[\d,]*(?:\.\d+)?", r["chosen"]["message"] or ""):
        v = _num(x)
        if abs(v - r["floor"]) <= 1 and not (r["buyer_offer"] is not None and abs(v - r["buyer_offer"]) <= 1):
            return True
    return False


def audit(pairs: list[dict]) -> list[tuple[str, bool, str]]:
    """Every check as (name, passed, detail)."""
    rows = [parse(p) for p in pairs]
    n = len(rows)
    out = []

    acc_c = sum(r["chosen"]["action"] == "accept" for r in rows)
    acc_r = sum(r["rejected"]["action"] == "accept" for r in rows)
    acc_only_c = sum(r["chosen"]["action"] == "accept" != r["rejected"]["action"] for r in rows)
    acc_only_r = sum(r["rejected"]["action"] == "accept" != r["chosen"]["action"] for r in rows)
    out.append(("accept is taught as good at least as often as bad", acc_only_c >= acc_only_r,
                f"accept only on the chosen side {acc_only_c}, only on the rejected side {acc_only_r} "
                f"(accept anywhere: chosen {acc_c}, rejected {acc_r})"))

    no_floor = sum(r["floor"] is None for r in rows)
    out.append(("every prompt states a floor", no_floor == 0, f"{no_floor} prompts without one"))

    bad_acc = sum(r["chosen"]["action"] == "accept" and r["floor"] is not None and r["buyer_offer"] is not None
                  and r["buyer_offer"] < r["floor"] for r in rows)
    bad_off = sum(r["chosen"]["action"] == "offer" and r["floor"] is not None
                  and (r["chosen"].get("price") or 0) < r["floor"] for r in rows)
    out.append(("no chosen reply goes below the floor", bad_acc + bad_off == 0,
                f"{bad_acc} chosen accepts of an offer below the floor, {bad_off} chosen offers below it"))

    leaks = sum(_states_floor(r) for r in rows)
    out.append(("no chosen reply names the floor (within $1, the metric's rule)", leaks == 0,
                f"{leaks} chosen replies do"))

    mismatch = sum(bool(SAYS_DEAL.search(r["chosen"]["message"] or "")) and r["chosen"]["action"] != "accept"
                   for r in rows)
    out.append(("chosen replies that say 'deal' also accept", mismatch == 0, f"{mismatch} say it without accepting"))

    types = Counter(r["type"] for r in rows)
    big = {t: c for t, c in types.items() if c / n > MAX_SHARE_ONE_TYPE}
    out.append((f"no pair type is over {MAX_SHARE_ONE_TYPE:.0%} of the file", not big,
                ", ".join(f"{t} {c} ({c / n:.0%})" for t, c in types.most_common())))

    for side in ("chosen", "rejected"):
        worst = []
        for t in types:
            texts = Counter(r[side]["message"] for r in rows if r["type"] == t)
            text, c = texts.most_common(1)[0]
            share = c / types[t]
            if share > MAX_SHARE_ONE_TEXT:
                worst.append(f"{t}: {share:.0%} are {text[:60]!r}")
        out.append((f"no {side} text repeats in over {MAX_SHARE_ONE_TEXT:.0%} of a pair type", not worst,
                    "; ".join(worst) or "ok"))
    return out


def action_table(pairs: list[dict]) -> str:
    """Chosen -> rejected action counts per pair type, the first thing to eyeball."""
    rows = [parse(p) for p in pairs]
    lines = []
    for t, c in Counter(r["type"] for r in rows).most_common():
        moves = Counter(f'{r["chosen"]["action"]}>{r["rejected"]["action"]}' for r in rows if r["type"] == t)
        lines.append(f"  {t:16} {c:5}  " + ", ".join(f"{m} {k}" for m, k in moves.most_common()))
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("path", help="DPO train.jsonl")
    pairs = [json.loads(line) for line in Path(p.parse_args().path).read_text(encoding="utf-8").splitlines() if line]
    print(f"{len(pairs)} pairs. chosen>rejected moves by pair type:\n{action_table(pairs)}\n")
    checks = audit(pairs)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    sys.exit(0 if all(ok for _, ok, _ in checks) else 1)


if __name__ == "__main__":
    main()
