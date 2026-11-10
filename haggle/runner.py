"""Experiment runner: executes the matrix in a config, within budget, and logs everything.

    uv run python -m haggle.runner configs/sanity.yaml
    uv run python -m haggle.runner configs/pilot.yaml --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml

from haggle.agents import sanity
from haggle.agents.base import Agent
from haggle.config import PROJECT_ROOT, check_models, config_hash, load_config, make_client, resolve
from haggle.engine import Game
from haggle.harnesses import ReplayBuyer, sample
from haggle.llm import CostTracker
from haggle.metrics import RIVAL_EVENT, format_summary, game_metrics, latest_run, load_results, summarize
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

    agent = AgenticSeller(ctx.client(spec["model"]), **_switches(spec, _AGENTIC_SWITCHES + _SELLER_SWITCHES))
    # Without this, honesty on/off (RQ4) would share one name and merge in every table.
    sw = _switches(spec, _SELLER_SWITCHES)
    defaults = {"honesty_instruction": True, "injection_defense": False}
    changed = [f"{k}={v}" for k, v in sorted(sw.items()) if v != defaults[k]]
    if changed:
        agent.name += f"[{','.join(changed)}]"
    return agent


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


@dataclass
class MarketJob:
    """One listing with several buyers competing for it (extension experiment)."""

    listing: Listing
    seed: int
    seller: dict
    buyers: list[dict]
    deadline_days: float
    floor_protection: bool
    buyers_see_competing_offers: bool = False

    def scenarios(self) -> list[Scenario]:
        return market_scenarios(self.listing, self.seed, self.buyers, self.deadline_days)


def market_scenarios(listing: Listing, seed: int, buyers: list[dict], deadline_days: float) -> list[Scenario]:
    """One seller means one floor; each buyer gets its own seeded budget and deadline."""
    floor = make_scenario(listing, seed).seller_floor
    asking = listing.asking_price
    out = []
    for i, buyer in enumerate(buyers, 1):
        rng = random.Random(f"{listing.id}:{seed}:b{i}")
        budget = max(round(floor + (asking - floor) * rng.uniform(0.3, 1.1)), floor + 1)
        days = buyer_deadline_days(buyer)
        sc = make_scenario(listing, seed, persona_label(buyer), seller_deadline_days=deadline_days,
                           buyer_deadline_days=days if days is not None else 5.0)
        out.append(sc.model_copy(update={"id": f"{sc.id}-b{i}", "buyer_budget": budget}))
    return out


def _seeds(spec) -> list[int]:
    if isinstance(spec, dict):
        return list(range(spec.get("start", 0), spec.get("start", 0) + spec["count"]))
    return list(spec)


def _market_jobs(cfg: dict, m: dict, protections: list) -> list[MarketJob]:
    mk = m["markets"]
    listings = load_listings(resolve(cfg["listings_path"]))
    jobs = []
    for seed in _seeds(m["seeds"]):
        chosen = sample(listings, mk["listing_sample"], hash_seed(f"market:{seed}"))
        for listing, seller, fp in itertools.product(chosen, m["sellers"], protections):
            for buyers in mk["buyer_sets"]:
                jobs.append(MarketJob(listing, seed, seller, buyers, mk.get("deadline_days", 7.0), fp,
                                      mk.get("buyers_see_competing_offers", False)))
    return jobs


def build_jobs(cfg: dict) -> list[Job | MarketJob]:
    m = cfg["matrix"]
    protections = m.get("floor_protection", [cfg["guardrails"]["floor_protection"]])
    if not isinstance(protections, list):
        protections = [protections]

    if "markets" in m:
        return _market_jobs(cfg, m, protections)

    if "replay" in m:
        src = m["replay"]
        # latest_run: runs/pilot may hold several timestamped runs, which must not be pooled.
        sources = [s for s in load_results(latest_run(resolve(src["source"]))) if s.outcome != "error"]
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


class MarketContextRecorder:
    """Logs the rival offer the platform showed the seller, as a guardrail event on the turn.

    The fake-vs-real competing offer comparison needs this, and an event keeps the frozen
    schema unchanged.
    """

    def __init__(self, agent: Agent):
        self.agent, self.name, self.model = agent, agent.name, agent.model

    def act(self, obs):
        reply = self.agent.act(obs)
        if obs.highest_competing_offer is None:
            return reply
        event = f"{RIVAL_EVENT}{obs.highest_competing_offer:.2f}"
        return reply.model_copy(update={"guardrail_events": [*reply.guardrail_events, event]})


def play_market(job: MarketJob, cfg: dict, cfg_hash: str, llm_log: Path | None, tracker: CostTracker | None = None) -> list[GameResult]:
    """Play one market. Returns one GameResult per buyer's chat."""
    from haggle.orchestrator import MultiGameOrchestrator

    scenarios = job.scenarios()
    orch = MultiGameOrchestrator(
        job.listing,
        deadline_days=job.deadline_days,
        enforce_floor_protection=job.floor_protection,
        enforce_price_direction=cfg["engine"].get("enforce_price_direction", False),
        max_turns_per_side=cfg["engine"]["max_turns_per_side"],
        buyers_see_competing_offers=job.buyers_see_competing_offers,
    )
    # One seller object across chats; stateful agents key their memory on session_id.
    seller = MarketContextRecorder(build_agent(SELLERS, job.seller, BuildContext(cfg, scenarios[0], llm_log, tracker)))
    for i, (scenario, spec) in enumerate(zip(scenarios, job.buyers), 1):
        buyer = build_agent(BUYERS, spec, BuildContext(cfg, scenario, llm_log, tracker))
        orch.add_session(f"b{i}", scenario, buyer, seller, arrival_hours=spec.get("arrival_hours", 0.0))
    try:
        orch.run(cfg_hash)
    except Exception as e:  # noqa: BLE001  one failed call must not kill a long experiment run
        for s in orch.sessions.values():
            if not s.game.is_over:
                s.game.outcome, s.game.error = "error", f"{type(e).__name__}: {e}"
    return [s.game.result(s.buyer, s.seller, cfg_hash) for s in orch.sessions.values()]


def _models_used(jobs: list) -> set[str]:
    specs = []
    for j in jobs:
        specs += [j.seller, *(j.buyers if isinstance(j, MarketJob) else [j.buyer])]
    return {k for s in specs for k in (s.get("model"), s.get("llm_model")) if k}


def _protections(cfg: dict) -> list:
    p = cfg["matrix"].get("floor_protection", [cfg["guardrails"]["floor_protection"]])
    return p if isinstance(p, list) else [p]


def _games_done(out: Path, jobs: list, cfg_hash: str) -> tuple[list[GameResult], int]:
    """The finished games of an interrupted run, checked against the jobs they claim to be, and how many jobs
    they cover.

    Jobs run in a fixed order and each finished job is flushed to games.jsonl, so the file holds the first jobs
    in order. A half-written last line (killed mid-write) is dropped and that game is played again.
    """
    done = []
    for line in (out / "games.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            done.append(GameResult.model_validate_json(line))
        except ValueError:
            break
    kept, i, n_jobs = [], 0, 0
    for job in jobs:
        n = len(job.buyers) if isinstance(job, MarketJob) else 1
        batch = done[i:i + n]
        if len(batch) < n:
            break  # a market is only done when all its chats are
        want = [s.id for s in job.scenarios()] if isinstance(job, MarketJob) else [job.scenario().id]
        if [r.scenario.id for r in batch] != want or any(r.config_hash != cfg_hash for r in batch):
            raise SystemExit(f"{out} doesn't match this config at game {i + 1}: start a fresh run instead.")
        kept += batch
        i += n
        n_jobs += 1
    return kept, n_jobs


def run(config_path: str | Path, out_root: Path | None = None, dry_run: bool = False,
        resume: str | Path | None = None) -> Path | None:
    """Play every job in the config. With resume=<run dir>, keep that run's finished games and play the rest,
    so a run stopped part-way (crash, low memory, closed laptop) doesn't have to start again."""
    cfg = load_config(config_path)
    jobs = build_jobs(cfg)
    budget = float(cfg["max_budget_usd"])
    n_games = sum(len(j.buyers) if isinstance(j, MarketJob) else 1 for j in jobs)
    print(f"{cfg['experiment_name']}: {n_games} games planned, budget ${budget:.2f}")
    if dry_run:
        est = cfg.get("est_cost_per_game_usd")
        if est is not None:
            print(f"  estimated spend: ${est * n_games:.2f} at ${est:.4f}/game (from config)")
        return None

    check_models(cfg, _models_used(jobs))
    cfg_hash = config_hash(cfg)
    if resume:
        out = Path(resume)
        results, skip = _games_done(out, jobs, cfg_hash)
    else:
        out = (out_root or PROJECT_ROOT / cfg.get("log_dir", "runs")) / cfg["experiment_name"] / datetime.now().strftime("%Y%m%d-%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        results, skip = [], 0
    # The resolved config (shared models merged in), so the run records what really ran.
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    llm_log = out / "llm_calls.jsonl"

    # Resumed games count towards the budget, so the cap still covers the whole run.
    spent = sum(t.cost_usd for r in results for t in r.turns)
    played, errors, aborted = len(results), 0, None
    tracker = CostTracker(budget)
    tracker.spent_usd = spent
    if resume:
        print(f"  resuming {out}: {played} games kept (${spent:.4f}), {len(jobs) - skip} jobs left")
    with (out / "games.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(r.model_dump_json() + "\n")
        f.flush()
        for i, job in enumerate(jobs):
            if i < skip:
                continue
            projected = spent + (spent / i if i else 0.0)  # per job: a market is several games
            if projected > budget:
                aborted = f"budget: spent ${spent:.4f}, next game projected to exceed ${budget:.2f}"
                break
            batch = (play_market(job, cfg, cfg_hash, llm_log, tracker) if isinstance(job, MarketJob)
                     else [play_job(job, cfg, cfg_hash, llm_log, tracker)])
            over_budget = None
            for result in batch:
                f.write(result.model_dump_json() + "\n")
                results.append(result)
                game_cost = sum(t.cost_usd for t in result.turns)
                spent += game_cost
                played += 1
                print(f"  [{i + 1}/{len(jobs)}] {result.scenario.id} {result.seller_agent} vs {result.buyer_agent}: "
                      f"{result.outcome} {result.deal_price or ''} (${game_cost:.4f})")
                if result.error and "BudgetExceeded" in result.error:
                    over_budget = result.error
            f.flush()
            errors = errors + 1 if any(r.outcome == "error" for r in batch) else 0
            if over_budget:
                aborted = f"budget: {over_budget}"
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
        "games_planned": n_games,
        "games_played": played,
        "error_games": sum(r.outcome == "error" for r in results),
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
    p.add_argument("--resume", metavar="RUN_DIR", help="finish an interrupted run of this config, keeping its games")
    args = p.parse_args(argv)
    run(args.config, dry_run=args.dry_run, resume=args.resume)


if __name__ == "__main__":
    main()
