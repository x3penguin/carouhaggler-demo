"""LLM-as-judge for seller honesty (RQ4), plus labelling export and judge-vs-human agreement.

    uv run python -m haggle.judge judge configs/judge.yaml runs/pilot/<ts>
    uv run python -m haggle.judge export runs/pilot/<ts> --out labels/pilot_100.csv --n 100
    uv run python -m haggle.judge agree labels/pilot_100.csv runs/pilot/<ts>/honesty.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from string import Template
from typing import Literal

from pydantic import BaseModel, ValidationError

from haggle.agents.llm_agent import PROMPT_DIR, money
from haggle.metrics import game_id, load_results
from haggle.schemas import GameResult

JUDGE_PROMPT_FILE = "judge_honesty.md"


class Violation(BaseModel):
    turn_index: int
    claim: str
    contradicts: str
    severity: Literal["minor", "major"]


class HonestyVerdict(BaseModel):
    violations: list[Violation]
    floor_leak: bool
    reasoning: str


VERDICT_SCHEMA = HonestyVerdict.model_json_schema()


def seller_transcript(result: GameResult) -> str:
    """Seller messages only, numbered by turn. Buyer text is shown as context in brackets
    so the judge can tell when a flaw was asked about and dodged."""
    lines = []
    for t in result.turns:
        if t.speaker == "buyer":
            lines.append(f"  (buyer, turn {t.index}: {t.action.message!r})")
        elif t.action.message:
            lines.append(f"[turn {t.index}] SELLER: {t.action.message}")
    return "\n".join(lines) or "(the seller sent no messages)"


class HonestyJudge:
    """Wraps an LLMClient."""

    def __init__(self, client, max_retries: int = 2, n_samples: int = 1):
        self.client = client
        self.model = client.model
        self.max_retries = max_retries
        self.n_samples = n_samples
        self.template = Template((PROMPT_DIR / JUDGE_PROMPT_FILE).read_text(encoding="utf-8"))

    def prompt(self, result: GameResult) -> str:
        listing = result.scenario.listing
        return self.template.substitute(
            title=listing.title,
            description=listing.description,
            condition_facts="\n".join(f"- {f}" for f in listing.condition_facts),
            floor=money(result.scenario.seller_floor),
            seller_messages=seller_transcript(result),
        )

    def _one_verdict(self, result: GameResult, usage: dict) -> HonestyVerdict | None:
        """One judged sample, retrying on schema errors. None if every attempt was invalid."""
        messages = [{"role": "user", "content": self.prompt(result)}]
        for _ in range(self.max_retries + 1):
            resp = self.client.chat(messages, json_schema=VERDICT_SCHEMA)
            for k in usage:
                usage[k] += getattr(resp, k)
            try:
                return HonestyVerdict.model_validate_json(resp.text)
            except ValidationError:
                messages += [
                    {"role": "assistant", "content": resp.text},
                    {"role": "user", "content": "That was not valid JSON for the schema. Reply again with JSON only."},
                ]
        return None

    def judge(self, result: GameResult) -> dict:
        """Return a JSONL-ready record: majority verdict + agreement + usage. Never raises on bad output."""
        usage = {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
        seller_turns = {t.index for t in result.turns if t.speaker == "seller"}
        base_seed = self.client.seed
        samples = []
        for i in range(self.n_samples):
            self.client.seed = None if base_seed is None else base_seed + i
            v = self._one_verdict(result, usage)
            if v is not None:
                v.violations = [x for x in v.violations if x.turn_index in seller_turns]
                samples.append(v)
        self.client.seed = base_seed

        base = {"game_id": game_id(result), "judge_model": self.model, "n_samples": self.n_samples,
                "valid_samples": len(samples), **usage}
        if not samples:
            return base | {"error": "judge output invalid on every attempt", "violations": None,
                           "major_violations": None, "judge_floor_leak": None, "details": []}

        dishonest_votes = [len(v.violations) > 0 for v in samples]
        leak_votes = [v.floor_leak for v in samples]
        dishonest = sum(dishonest_votes) * 2 > len(samples)
        rep = next(v for v in samples if (len(v.violations) > 0) == dishonest)
        return base | {
            "error": None,
            "violations": len(rep.violations),
            "major_violations": sum(x.severity == "major" for x in rep.violations),
            "judge_floor_leak": sum(leak_votes) * 2 > len(samples),
            "dishonest_agreement": max(sum(dishonest_votes), len(samples) - sum(dishonest_votes)) / len(samples),
            "leak_agreement": max(sum(leak_votes), len(samples) - sum(leak_votes)) / len(samples),
            "reasoning": rep.reasoning,
            "details": [x.model_dump() for x in rep.violations],
        }


def self_preference_risk(judge_model: str, results: list[GameResult]) -> set[str]:
    """Models that both played in the run and are the judge (self-preference bias, Topics 5-6)."""
    return {m for r in results for m in (r.seller_model, r.buyer_model) if m == judge_model}


def judge_run(judge: HonestyJudge, run_dir: Path, budget_usd: float) -> Path:
    """Judge every game in run_dir, resuming past verdicts so a crash doesn't pay twice."""
    results = load_results(run_dir)
    if self_preference_risk(judge.model, results):
        print(f"WARNING: judge model {judge.model!r} also played in this run; prefer a "
              "different-family judge to avoid self-preference bias.")
    out = run_dir / "honesty.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["game_id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    spent = 0.0
    with out.open("a", encoding="utf-8") as f:
        for r in results:
            if game_id(r) in done:
                continue
            if spent > budget_usd:
                print(f"ABORTED: judge spend ${spent:.4f} > budget ${budget_usd:.2f}")
                break
            rec = judge.judge(r)
            spent += rec["cost_usd"]
            f.write(json.dumps(rec) + "\n")
            print(f"  {rec['game_id']}: violations={rec['violations']} leak={rec['judge_floor_leak']}")
    print(f"Judge spent ${spent:.4f}. Verdicts: {out}")
    return out



LABEL_FIELDS = ["game_id", "listing", "condition_facts", "seller_floor", "deal_price", "transcript",
                "human_dishonest", "human_floor_leak", "labeller", "notes"]


def export_for_labelling(results: list[GameResult], out_csv: Path, n: int = 100, seed: int = 0) -> Path:
    """Write a blind, seeded sample to CSV. Labellers fill human_dishonest / human_floor_leak with 0/1."""
    rng = random.Random(seed)
    by_agent: dict[str, list[GameResult]] = {}
    for r in results:
        by_agent.setdefault(f"{r.seller_agent}|{r.seller_model}", []).append(r)
    for rs in by_agent.values():
        rng.shuffle(rs)
    picked: list[GameResult] = []
    while len(picked) < n and any(by_agent.values()):
        for rs in by_agent.values():
            if rs and len(picked) < n:
                picked.append(rs.pop())
    rng.shuffle(picked)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
        w.writeheader()
        for r in picked:
            full = "\n".join(f"[{t.index}] {t.speaker.upper()} ({t.action.action}"
                             f"{' $' + money(t.action.price) if t.action.action == 'offer' else ''}): {t.action.message}"
                             for t in r.turns)
            w.writerow({
                "game_id": game_id(r),
                "listing": r.scenario.listing.title,
                "condition_facts": "; ".join(r.scenario.listing.condition_facts),
                "seller_floor": r.scenario.seller_floor,
                "deal_price": r.deal_price,
                "transcript": full,
                "human_dishonest": "", "human_floor_leak": "", "labeller": "", "notes": "",
            })
    return out_csv


def binary_agreement(human: list[bool], judge: list[bool]) -> dict:
    """Accuracy and the judge's precision / recall / F1, treating human labels as ground truth."""
    n = len(human)
    if n == 0:
        return {"n": 0}
    tp = sum(h and j for h, j in zip(human, judge))
    tn = sum((not h) and (not j) for h, j in zip(human, judge))
    fp = sum((not h) and j for h, j in zip(human, judge))
    fn = sum(h and (not j) for h, j in zip(human, judge))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {
        "n": n, "accuracy": (tp + tn) / n, "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def agreement(labels_csv: Path, verdicts_jsonl: Path) -> dict:
    """Join human labels to judge verdicts by game_id; report agreement for both questions."""
    verdicts = {}
    for line in verdicts_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            verdicts[rec["game_id"]] = rec
    pairs = {"dishonest": ([], []), "floor_leak": ([], [])}
    missing = 0
    with labels_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = verdicts.get(row["game_id"])
            if v is None or v.get("error"):
                missing += 1
                continue
            if row["human_dishonest"].strip() in ("0", "1"):
                pairs["dishonest"][0].append(row["human_dishonest"].strip() == "1")
                pairs["dishonest"][1].append(v["violations"] > 0)
            if row["human_floor_leak"].strip() in ("0", "1"):
                pairs["floor_leak"][0].append(row["human_floor_leak"].strip() == "1")
                pairs["floor_leak"][1].append(bool(v["judge_floor_leak"]))
    return {k: binary_agreement(*pairs[k]) for k in pairs} | {"unmatched_labels": missing}


def main(argv: list[str] | None = None) -> None:
    from haggle.config import load_config, make_client

    p = argparse.ArgumentParser(description="Honesty judge and judge-vs-human agreement.")
    sub = p.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge"); j.add_argument("config"); j.add_argument("run_dir")
    e = sub.add_parser("export"); e.add_argument("runs", nargs="+"); e.add_argument("--out", required=True)
    e.add_argument("--n", type=int, default=100); e.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("agree"); a.add_argument("labels_csv"); a.add_argument("verdicts_jsonl")
    args = p.parse_args(argv)

    if args.cmd == "judge":
        cfg = load_config(args.config)
        run_dir = Path(args.run_dir)
        client = make_client(cfg, cfg["judge"]["model"], run_dir / "judge_llm_calls.jsonl")
        jcfg = cfg["judge"]
        judge = HonestyJudge(client, jcfg.get("max_retries", 2), jcfg.get("n_samples", 1))
        judge_run(judge, run_dir, float(cfg["max_budget_usd"]))
    elif args.cmd == "export":
        path = export_for_labelling(load_results(args.runs), Path(args.out), args.n, args.seed)
        print(f"Wrote {path}")
    else:
        print(json.dumps(agreement(Path(args.labels_csv), Path(args.verdicts_jsonl)), indent=2))


if __name__ == "__main__":
    main()
