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
ASKING = re.compile(r"Asking price: SGD ([\d,.]+)")
BUYER_OFFER = re.compile(r"their current offer \$([\d,.]+)")
# Words that tell the buyer the deal is done. If the action isn't `accept`, the chat and the move disagree.
SAYS_DEAL = re.compile(r"^\W*(ok(ay)?[, ]+)?(deal|it'?s a deal|sold|done|agreed)\b", re.IGNORECASE)

MAX_SHARE_ONE_TEXT = 0.05   # no single reply text may be more than 5% of one side of a pair type
MIN_TYPE_FOR_SHARE = 50     # ...checked only for pair types this big: in 16 pairs one repeat is already 6%
MAX_SHARE_ONE_TYPE = 0.40   # no pair type may be more than 40% of the file
ACCEPT_BALANCE = (0.8, 1.25)  # accept-wins / accept-losses: v2 had 0, v3 had 2.6, both went wrong
SHORTCUT_BAND = (0.60, 0.95)  # offers (share of asking) where the floor, not the offer size, must decide
MAX_SHORTCUT_ACCURACY = 0.65  # how well offer size alone may predict "accept wins" inside that band
CONTAMINATION_WORDS = 6       # a run of this many words shared with an attack script counts as copied


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def parse(pair: dict) -> dict:
    """The facts each check needs: floor, the buyer's standing offer, and both replies."""
    system, last = pair["messages"][0]["content"], pair["messages"][-1]["content"]
    floor, offer, asking = FLOOR.search(system), BUYER_OFFER.findall(last), ASKING.search(system)
    return {"type": pair["id"].rsplit("_", 1)[-1],
            "floor": _num(floor.group(1)) if floor else None,
            "asking": _num(asking.group(1)) if asking else None,
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


def _shortcut_accuracy(rows: list[dict]) -> tuple[float, int]:
    """Among pairs where exactly one side accepts and the offer is inside SHORTCUT_BAND, the best accuracy a single
    threshold on offer/asking gets at predicting "accept is the chosen side". 0.5 = offer size tells nothing (only the
    floor decides); near 1.0 = the model can skip the floor. v3's close and subfloor offers barely overlapped."""
    pts = sorted((r["buyer_offer"] / r["asking"], r["chosen"]["action"] == "accept") for r in rows
                 if (r["chosen"]["action"] == "accept") != (r["rejected"]["action"] == "accept")
                 and r["buyer_offer"] and r["asking"] and SHORTCUT_BAND[0] <= r["buyer_offer"] / r["asking"] <= SHORTCUT_BAND[1])
    n, pos = len(pts), sum(y for _, y in pts)
    if n == 0:
        return 0.5, 0
    best, below_pos = max(pos, n - pos), 0  # threshold below every point
    for i, (_, y) in enumerate(pts, 1):    # threshold just above point i: the first i are "below"
        below_pos += y
        right = (i - below_pos) + (pos - below_pos)  # below -> refuse, above -> accept
        best = max(best, right, n - right)
    return best / n, n


def _ngrams(text: str, n: int) -> set:
    words = re.findall(r"[a-z']+|<p>", re.sub(r"\$\{p\}|\$?\d[\d,.]*", " <p> ", text.lower()))
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def contamination(pairs: list[dict], attacks_path: Path, source_path: Path | None = None) -> list[str]:
    """Buyer text in the pairs that shares a CONTAMINATION_WORDS-word run with an attack script, i.e. training on the test.

    Runs that also occur in the buyers' messages of source_path (the CraigslistBargain SFT data) are allowed: those were
    written by the dataset's crowdworkers in 2017, so a shared phrase such as "could you let it go for $X" is ordinary
    haggling, not a copy of our scripts. Anything we wrote ourselves (pressure lines, openers) gets no such pass."""
    if not attacks_path.exists():
        return []
    test = set()
    for script in json.loads(attacks_path.read_text(encoding="utf-8")):
        for step in script["steps"]:
            test |= _ngrams(step["message"], CONTAMINATION_WORDS)
    if source_path is not None and source_path.exists():
        for line in source_path.read_text(encoding="utf-8").splitlines():
            for m in json.loads(line)["messages"][1:]:
                if m["role"] == "user":
                    test -= _ngrams(m["content"], CONTAMINATION_WORDS)
    hits = []
    for p in pairs:
        for m in p["messages"][1:]:
            if m["role"] == "user" and (shared := _ngrams(m["content"], CONTAMINATION_WORDS) & test):
                hits.append(f'{p["id"]}: "{sorted(shared)[0]}"')
                break
    return hits


def audit(pairs: list[dict], attacks_path: Path = Path("data/attacks.json"),
          source_path: Path | None = Path("data/craigslist/sft/train.jsonl")) -> list[tuple[str, bool, str]]:
    """Every check as (name, passed, detail)."""
    rows = [parse(p) for p in pairs]
    n = len(rows)
    out = []

    acc_c = sum(r["chosen"]["action"] == "accept" for r in rows)
    acc_r = sum(r["rejected"]["action"] == "accept" for r in rows)
    acc_only_c = sum(r["chosen"]["action"] == "accept" != r["rejected"]["action"] for r in rows)
    acc_only_r = sum(r["rejected"]["action"] == "accept" != r["chosen"]["action"] for r in rows)
    ratio = acc_only_c / acc_only_r if acc_only_r else float("inf")
    out.append((f"accept wins and loses about equally often (ratio {ACCEPT_BALANCE[0]}-{ACCEPT_BALANCE[1]})",
                ACCEPT_BALANCE[0] <= ratio <= ACCEPT_BALANCE[1],
                f"accept only on the chosen side {acc_only_c}, only on the rejected side {acc_only_r}, ratio {ratio:.2f} "
                f"(accept anywhere: chosen {acc_c}, rejected {acc_r})"))

    fair_refused = sum(r["rejected"]["action"] == "accept" != r["chosen"]["action"] and r["floor"] is not None
                       and r["buyer_offer"] is not None and r["buyer_offer"] >= r["floor"] for r in rows)
    out.append(("accept loses only to offers below the floor", fair_refused == 0,
                f"{fair_refused} pairs teach refusing an offer at or above the floor"))

    acc, n_band = _shortcut_accuracy(rows)
    out.append((f"offer size alone can't tell accept from refuse (offers {SHORTCUT_BAND[0]:.0%}-{SHORTCUT_BAND[1]:.0%} of asking)",
                acc <= MAX_SHORTCUT_ACCURACY,
                f"best threshold on offer/asking is right {acc:.0%} of the time over {n_band} pairs (max {MAX_SHORTCUT_ACCURACY:.0%})"))

    hits = contamination(pairs, attacks_path, source_path)
    out.append((f"no buyer text copies an attack script ({CONTAMINATION_WORDS}-word runs)", not hits,
                (f"{len(hits)} pairs, e.g. " + "; ".join(hits[:3]) if hits else f"ok (checked against {attacks_path})")
                + ("" if source_path is None or source_path.exists() else
                   f" [{source_path} not found, so CraigslistBargain's own phrases aren't exempt: run "
                   "`uv run python -m finetune.prepare_data` first]")))

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
        for t in (t for t, c in types.items() if c >= MIN_TYPE_FOR_SHARE):
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
