"""Gate 1 for a DPO model: on held-out floor twins, does it accept when the offer clears the floor and refuse when it doesn't?

Run from the repo root (local models through Ollama, so $0):
    uv run python -m finetune.check_decisions finetune/dpo/v4/valid.jsonl --models finetuned finetuned_dpo finetuned_dpo_v3

A twin is one prompt shown twice with only the floor changed (see build_dpo_pairs_v4.py), so the
floor is the only thing that should change the answer. Per model this reports how often it
accepts on each side, and how often it gets *both* halves of a twin right. A model that ignores
the floor accepts both or neither, and scores at most about 50% on "both right" whatever its
accept rate: v2's "never accept" and v3's "accept readily" both fail it. Each prompt is sent the
way the live seller sends it (same messages, the Action JSON schema, the model's temperature and
the config seed), so this is a cheap preview of the closing check and the attack run.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from haggle.config import load_config, make_client
from haggle.schemas import Action

FAMILIES = {"twin": ("hold", "close"), "pressure": ("pressure-hold", "pressure-close"),
            "opener": ("opener-hold", "opener-close")}
SAYS_DEAL = re.compile(r"^\W*(ok(ay)?[, ]+)?(deal|it'?s a deal|sold|done|agreed)\b", re.IGNORECASE)


def twins(pairs: list[dict], per_family: int, seed: int) -> dict:
    """{family: [(hold pair, close pair), ...]}, a seeded sample of complete twins."""
    by = defaultdict(dict)
    for p in pairs:
        key, kind = p["id"].rsplit("_", 1)
        by[key][kind] = p
    rng, out = random.Random(seed), {}
    for fam, (h, c) in FAMILIES.items():
        found = [(by[k][h], by[k][c]) for k in sorted(by) if h in by[k] and c in by[k]]
        rng.shuffle(found)
        out[fam] = found[:per_family]
    return out


def decide(client, messages: list[dict]) -> dict:
    """The model's move on one prompt, called the way PromptedAgent.call_json calls it."""
    resp = client.chat(messages, json_schema=Action.model_json_schema())
    try:
        a = Action.model_validate_json(resp.text)
    except ValueError:
        return {"action": None, "says_deal": False}
    return {"action": a.action, "says_deal": bool(SAYS_DEAL.search(a.message or "")) and a.action != "accept"}


def check(client, sample: dict) -> dict:
    res = {}
    for fam, pairs in sample.items():
        rows = [(decide(client, h["messages"]), decide(client, c["messages"])) for h, c in pairs]
        n = len(rows) or 1
        res[fam] = {"n": len(rows),
                    "accepts_below_floor": sum(h["action"] == "accept" for h, _ in rows) / n,
                    "accepts_above_floor": sum(c["action"] == "accept" for _, c in rows) / n,
                    "both_right": sum(h["action"] != "accept" and c["action"] == "accept" for h, c in rows) / n,
                    "says_deal_no_accept": sum(x["says_deal"] for r in rows for x in r) / (2 * n),
                    "bad_json": sum(x["action"] is None for r in rows for x in r) / (2 * n)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pairs", type=Path, help="held-out pairs from build_dpo_pairs_v4 (valid.jsonl)")
    ap.add_argument("--models", nargs="+", required=True, help="keys in configs/models.yaml")
    ap.add_argument("--config", default="configs/closing_check.yaml", help="supplies models_file and seed")
    ap.add_argument("--per-family", type=int, default=25)
    ap.add_argument("--out", type=Path, help="write the results as JSON here too")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pairs = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line]
    sample = twins(pairs, args.per_family, cfg["seed"])
    results = {}
    for key in args.models:
        results[key] = check(make_client(cfg, key), sample)
        print(f"\n{key} ({cfg['models'][key]['model']})")
        print(f"  {'family':9} {'twins':>5} {'accept<floor':>12} {'accept>=floor':>13} {'both right':>10} "
              f"{'deal,no accept':>14} {'bad JSON':>8}")
        for fam, r in results[key].items():
            print(f"  {fam:9} {r['n']:5} {r['accepts_below_floor']:12.0%} {r['accepts_above_floor']:13.0%} "
                  f"{r['both_right']:10.0%} {r['says_deal_no_accept']:14.0%} {r['bad_json']:8.0%}", flush=True)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
