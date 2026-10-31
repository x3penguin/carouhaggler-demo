"""Experiment runner: executes the matrix in a config, within budget, and logs everything.

    uv run python -m haggle.runner configs/sanity.yaml
    uv run python -m haggle.runner configs/pilot.yaml --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from haggle.agents import sanity
from haggle.agents.base import Agent
from haggle.config import PROJECT_ROOT, config_hash, load_config, make_client, resolve
from haggle.engine import Game
from haggle.harnesses import ReplayBuyer, sample
from haggle.llm import CostTracker
from haggle.metrics import format_summary, game_metrics, load_results, summarize
from haggle.scenarios import load_listings, make_scenario
from haggle.schemas import GameResult, Listing, Scenario



@dataclass
class BuildContext:
    cfg: dict
    scenario: Scenario
    llm_log: Path | None
    tracker: CostTracker | None = None
    client: Callable[[str], object] = field(init=False)

    def __post_init__(self):
        self.client = lambda model_key: make_client(self.cfg, model_key, self.llm_log, self.tracker)


_SELLER_SWITCHES = ("honesty_instruction", "injection_defense")
_AGENTIC_SWITCHES = ("use_tools", "use_planning", "use_reflection")
_BUYER_SWITCHES = ("probe_condition", "injection_defense")


def _switches(spec: dict, names: tuple[str, ...]) -> dict:
    return {k: spec[k] for k in names if k in spec}


def _prompted_seller(spec: dict, ctx: BuildContext) -> Agent:
    from haggle.agents.prompted_seller import PromptedSeller

    sw = _switches(spec, _SELLER_SWITCHES)
    defaults = {"honesty_instruction": True, "injection_defense": False}
    changed = [f"{k}={v}" for k, v in sorted(sw.items()) if v != defaults[k]]
    name = "prompted_seller" + (f"[{','.join(changed)}]" if changed else "")
    return PromptedSeller(ctx.client(spec["model"]), name=name, **sw)


def _agentic_seller(spec: dict, ctx: BuildContext) -> Agent:
    from haggle.agents.agentic import AgenticSeller

    return AgenticSeller(ctx.client(spec["model"]), **_switches(spec, _AGENTIC_SWITCHES + _SELLER_SWITCHES))


def _llm_buyer(spec: dict, ctx: BuildContext) -> Agent:
    from haggle.agents.buyers import LLMBuyer

    return LLMBuyer(ctx.client(spec["model"]), spec.get("persona", "fair"), **_switches(spec, _BUYER_SWITCHES))


def _agentic_buyer(spec: dict, ctx: BuildContext) -> Agent:
    from haggle.agents.agentic import AgenticBuyer

    return AgenticBuyer(ctx.client(spec["model"]), spec.get("persona", "fair"),
                        **_switches(spec, _AGENTIC_SWITCHES + _BUYER_SWITCHES))


def _attack_buyer(spec: dict, ctx: BuildContext) -> Agent:
    from haggle.agents.attackers import AttackBuyer, load_attacks

    script = next(a for a in load_attacks() if a.id == spec["script"])
    client = ctx.client(spec["llm_model"]) if spec.get("llm_model") else None
    return AttackBuyer(script, client=client, twin=spec.get("twin", False))


_S = sanity
SELLERS: dict[str, Callable[[dict, BuildContext], Agent]] = {
    "prompted_seller": _prompted_seller,
    "agentic_seller": _agentic_seller,
    "sanity:accept_above_floor": lambda spec, ctx: _S.AcceptAboveFloorSeller(),
    "sanity:stubborn": lambda spec, ctx: _S.StubbornSeller(),
    "sanity:leaky": lambda spec, ctx: _S.LeakySeller(),
    "sanity:below_floor": lambda spec, ctx: _S.BelowFloorSeller(),
    "sanity:ghosting": lambda spec, ctx: _S.GhostingSeller(ctx.scenario, **spec.get("kwargs", {})),
}
BUYERS: dict[str, Callable[[dict, BuildContext], Agent]] = {
    "llm_buyer": _llm_buyer,
    "agentic_buyer": _agentic_buyer,
    "attack": _attack_buyer,
    "sanity:full_budget": lambda spec, ctx: _S.FullBudgetBuyer(ctx.scenario),
    "sanity:floor": lambda spec, ctx: _S.FloorBuyer(ctx.scenario),
    "sanity:linear": lambda spec, ctx: _S.LinearConcessionBuyer(ctx.scenario, **spec.get("kwargs", {})),
    "sanity:delayed": lambda spec, ctx: _S.DelayedBuyer(ctx.scenario, **spec.get("kwargs", {})),
}


def build_agent(registry: dict, spec: dict, ctx: BuildContext) -> Agent:
    agent = registry[spec["agent"]](spec, ctx)
    if spec.get("guard"):
        from haggle.guardrails import GuardedAgent

        agent = GuardedAgent(agent, **spec["guard"])
    return agent


def expand_buyers(buyers: list[dict]) -> list[dict]:
    """Turn `{agent: attacks, ...}` entries into one buyer spec per script (and twin)."""
    out = []
    for b in buyers:
        if b["agent"] != "attacks":
            out.append(b)
            continue
        from haggle.agents.attackers import load_attacks

        cats = b.get("categories", "all")
        for script in load_attacks():
            if cats != "all" and script.category not in cats:
                continue
            for twin in ([False, True] if b.get("twins", True) else [False]):
                kind = "twin" if twin else "attack"
                out.append({"agent": "attack", "script": script.id, "twin": twin,
                            "persona": f"{kind}:{script.category}", "llm_model": b.get("llm_model")})
    return out


def persona_label(buyer_spec: dict) -> str:
    """What goes into Scenario.buyer_persona; metrics read attack/twin categories from it."""
    if "persona" in buyer_spec:
        return buyer_spec["persona"]
    return buyer_spec["agent"]


def buyer_deadline_days(spec: dict) -> float | None:
    from haggle.agents.buyers import BUYER_DEADLINES

    d = spec.get("deadline")
    return BUYER_DEADLINES[d] if isinstance(d, str) else d




@dataclass
class Job:
    """One game to play. Ordering of jobs is deterministic, so runs are reproducible."""

    listing: Listing
    seed: int
    seller: dict
    buyer: dict
    floor_protection: bool
    source: GameResult | None = None

    def scenario(self) -> Scenario:
        if self.source is not None:
            return self.source.scenario
        days = buyer_deadline_days(self.buyer)
        kwargs = {"buyer_deadline_days": days} if days is not None else {}
        return make_scenario(self.listing, self.seed, persona_label(self.buyer), **kwargs)


def _seeds(spec) -> list[int]:
    if isinstance(spec, dict):
        return list(range(spec.get("start", 0), spec.get("start", 0) + spec["count"]))
    return list(spec)


def build_jobs(cfg: dict) -> list[Job]:
    m = cfg["matrix"]
    protections = m.get("floor_protection", [cfg["guardrails"]["floor_protection"]])
    if not isinstance(protections, list):
        protections = [protections]

    if "replay" in m:
        src = m["replay"]
        sources = load_results(resolve(src["source"]))
        sources = sample(sources, src.get("max_games", len(sources)), cfg["seed"])
        return [
            Job(s.scenario.listing, s.scenario.seed, seller, {"agent": "replay"}, fp, source=s)
            for s, seller, fp in itertools.product(sources, m["sellers"], protections)
        ]

    listings = load_listings(resolve(cfg["listings_path"]))
    if m.get("listings", "all") != "all":
        wanted = set(m["listings"])
        listings = [l for l in listings if l.id in wanted]
    jobs = []
    buyers = expand_buyers(m["buyers"])
    k = m.get("listing_sample")
    for seed in _seeds(m["seeds"]):
        for buyer in buyers:
            for listing in listings if k is None else _listings_for(buyer, listings, k, seed):
                for seller, fp in itertools.product(m["sellers"], protections):
                    b = {**buyer, "model": seller["model"]} if m.get("self_play") else buyer
                    jobs.append(Job(listing, seed, seller, b, fp))
    return jobs[: cfg.get("max_games", len(jobs))]


def _listings_for(buyer: dict, listings: list[Listing], k: int, seed: int) -> list[Listing]:
    """Seeded sample of k listings for one buyer. Keyed on the attack script id when there is
    one, so an attack and its twin always face the same listings (that's what makes the
    attack − twin difference a fair comparison)."""
    key = buyer.get("script") or persona_label(buyer)
    return sample(listings, k, hash_seed(f"{key}:{seed}"))


def hash_seed(text: str) -> int:
    """Stable across runs (Python's hash() is salted per process, so it can't be used)."""
    import hashlib

    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)




def play_job(job: Job, cfg: dict, cfg_hash: str, llm_log: Path | None, tracker: CostTracker | None = None) -> GameResult:
    scenario = job.scenario()
    ctx = BuildContext(cfg, scenario, llm_log, tracker)
    seller = build_agent(SELLERS, job.seller, ctx)
    buyer = ReplayBuyer(job.source) if job.source is not None else build_agent(BUYERS, job.buyer, ctx)
    eng = cfg["engine"]
    game = Game(
        scenario,
        max_turns_per_side=eng["max_turns_per_side"],
        max_parse_retries=eng["max_parse_retries"],
        floor_protection=job.floor_protection,
        budget_protection=True,
        enforce_price_direction=eng.get("enforce_price_direction", False),
    )
    try:
        game.play(buyer, seller)
    except Exception:  # noqa: BLE001
        game.outcome = "error"
        game.error = traceback.format_exc(limit=1)
    if game.error and game.outcome == "error":
        print(f"  ! game error on {scenario.id}: {game.error}", file=sys.stderr)
    result = game.result(buyer, seller, cfg_hash)
    if len(set(_protections(cfg))) > 1:
        result.seller_agent += f"+fp{int(job.floor_protection)}"
    return result


def _protections(cfg: dict) -> list:
    p = cfg["matrix"].get("floor_protection", [cfg["guardrails"]["floor_protection"]])
    return p if isinstance(p, list) else [p]


def run(config_path: str | Path, out_root: Path | None = None, dry_run: bool = False) -> Path | None:
    cfg = load_config(config_path)
    jobs = build_jobs(cfg)
    budget = float(cfg["max_budget_usd"])
    print(f"{cfg['experiment_name']}: {len(jobs)} games planned, budget ${budget:.2f}")
    if dry_run:
        est = cfg.get("est_cost_per_game_usd")
        if est is not None:
            print(f"  estimated spend: ${est * len(jobs):.2f} at ${est:.4f}/game (from config)")
        return None

    cfg_hash = config_hash(cfg)
    out = (out_root or PROJECT_ROOT / cfg.get("log_dir", "runs")) / cfg["experiment_name"] / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(resolve(config_path), out / "config.yaml")
    llm_log = out / "llm_calls.jsonl"
    tracker = CostTracker(budget)

    spent, played, errors, aborted = 0.0, 0, 0, None
    results: list[GameResult] = []
    with (out / "games.jsonl").open("w", encoding="utf-8") as f:
        for i, job in enumerate(jobs):
            projected = spent + (spent / played if played else 0.0)
            if projected > budget:
                aborted = f"budget: spent ${spent:.4f}, next game projected to exceed ${budget:.2f}"
                break
            result = play_job(job, cfg, cfg_hash, llm_log, tracker)
            f.write(result.model_dump_json() + "\n")
            f.flush()
            results.append(result)
            game_cost = sum(t.cost_usd for t in result.turns)
            spent += game_cost
            played += 1
            errors = errors + 1 if result.outcome == "error" else 0
            print(f"  [{i + 1}/{len(jobs)}] {result.scenario.id} {result.seller_agent} vs {result.buyer_agent}: "
                  f"{result.outcome} {result.deal_price or ''} (${game_cost:.4f})")
            if result.error and "BudgetExceeded" in result.error:
                aborted = f"budget: {result.error}"
                break
            if errors >= cfg.get("max_consecutive_errors", 3):
                aborted = f"{errors} consecutive game errors (is the model server up?)"
                break
            if spent > budget:
                aborted = f"budget: spent ${spent:.4f} > ${budget:.2f}"
                break

    write_summary(out, results, cfg)
    manifest = {
        "experiment": cfg["experiment_name"],
        "config_hash": cfg_hash,
        "games_planned": len(jobs),
        "games_played": played,
        "spent_usd": spent,
        "cost_per_game_usd": spent / played if played else None,
        "budget_usd": budget,
        "aborted": aborted,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if aborted:
        print(f"ABORTED: {aborted}")
    print(f"Spent ${spent:.4f} over {played} games (${manifest['cost_per_game_usd'] or 0:.4f}/game). Logs: {out}")
    return out


def write_summary(out: Path, results: list[GameResult], cfg: dict) -> list[dict]:
    rows = [game_metrics(r) for r in results]
    group_by = tuple(cfg.get("summary_group_by", ["seller_agent", "seller_model", "buyer_persona"]))
    boot = cfg.get("bootstrap", {})
    summaries = summarize(rows, group_by=group_by, n_boot=boot.get("n_boot", 2000), seed=boot.get("seed", cfg["seed"]))
    (out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    table = format_summary(summaries, group_by)
    (out / "summary.txt").write_text(table + "\n", encoding="utf-8")
    print("\n" + table + "\n")
    return summaries


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("config", help="path to configs/<experiment>.yaml")
    p.add_argument("--dry-run", action="store_true", help="print the matrix size, play nothing")
    args = p.parse_args(argv)
    run(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
