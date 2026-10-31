"""Runner tests: matrix expansion, logging layout, budget abort, replay mode. No LLM calls."""

import json

import pytest
import yaml

from haggle import runner
from haggle.agents.base import AgentReply
from haggle.metrics import load_results
from haggle.schemas import Action

THREE = ["phone-001", "console-001", "furniture-001"]

BASE = {
    "seed": 1,
    "max_budget_usd": 0.0,
    "listings_path": "data/listings_sg.json",
    "engine": {"max_turns_per_side": 4, "max_parse_retries": 1},
    "guardrails": {"floor_protection": True},
    "bootstrap": {"n_boot": 100, "seed": 0},
    "models": {},
}


def write_cfg(tmp_path, name, **overrides):
    cfg = {**BASE, "experiment_name": name, **overrides}
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_sanity_run_writes_all_artifacts_and_is_reproducible(tmp_path):
    matrix = {"listings": THREE, "seeds": [0, 1],
              "sellers": [{"agent": "sanity:accept_above_floor"}],
              "buyers": [{"agent": "sanity:full_budget"}, {"agent": "sanity:floor"}]}
    cfg = write_cfg(tmp_path, "t", matrix=matrix)
    out = runner.run(cfg, out_root=tmp_path / "runs")
    for f in ["config.yaml", "games.jsonl", "summary.json", "summary.txt", "manifest.json"]:
        assert (out / f).exists()
    results = load_results(out)
    assert len(results) == 2 * 3 * 2
    summary = {s["buyer_persona"]: s for s in json.loads((out / "summary.json").read_text())}
    assert summary["sanity:full_budget"]["surplus_share"] == pytest.approx(1.0)
    assert summary["sanity:floor"]["surplus_share"] == pytest.approx(0.0)

    out2 = runner.run(cfg, out_root=tmp_path / "runs2")
    again = load_results(out2)
    assert [r.model_dump() for r in results] == [r.model_dump() for r in again]


def test_max_games_and_seed_major_order(tmp_path):
    cfg = {**BASE, "experiment_name": "x", "max_games": 4,
           "matrix": {"listings": THREE, "seeds": {"start": 0, "count": 3},
                      "sellers": [{"agent": "sanity:stubborn"}], "buyers": [{"agent": "sanity:linear"}]}}
    jobs = runner.build_jobs(cfg)
    assert len(jobs) == 4 and [j.seed for j in jobs] == [0, 0, 0, 1]


def test_self_play_copies_seller_model_to_buyer():
    cfg = {**BASE, "experiment_name": "x",
           "matrix": {"listings": "all", "seeds": [0], "self_play": True,
                      "sellers": [{"agent": "prompted_seller", "model": "m1"}],
                      "buyers": [{"agent": "llm_buyer", "persona": "fair"}]}}
    assert all(j.buyer["model"] == "m1" for j in runner.build_jobs(cfg))


class CostlySeller:
    name, model = "costly", "m"

    def act(self, obs):
        return AgentReply(action=Action(message="", action="offer", price=obs.listing.asking_price), cost_usd=0.01)


def test_budget_abort(tmp_path, monkeypatch):
    monkeypatch.setitem(runner.SELLERS, "costly", lambda spec, ctx: CostlySeller())
    matrix = {"listings": THREE, "seeds": {"start": 0, "count": 10},
              "sellers": [{"agent": "costly"}], "buyers": [{"agent": "sanity:floor"}]}
    cfg = write_cfg(tmp_path, "b", matrix=matrix, max_budget_usd=0.10)
    out = runner.run(cfg, out_root=tmp_path / "runs")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["games_played"] == 2 and manifest["aborted"].startswith("budget")
    assert manifest["spent_usd"] <= 0.10
    assert manifest["cost_per_game_usd"] == pytest.approx(0.04)


def test_replay_mode_uses_source_scenarios(tmp_path):
    src_matrix = {"listings": THREE, "seeds": [0],
                  "sellers": [{"agent": "sanity:accept_above_floor"}], "buyers": [{"agent": "sanity:full_budget"}]}
    src = runner.run(write_cfg(tmp_path, "src", matrix=src_matrix), out_root=tmp_path / "runs")
    replay_matrix = {"replay": {"source": str(src)},
                     "sellers": [{"agent": "sanity:stubborn"}, {"agent": "sanity:accept_above_floor"}]}
    out = runner.run(write_cfg(tmp_path, "rep", matrix=replay_matrix), out_root=tmp_path / "runs")
    results, sources = load_results(out), load_results(src)
    assert len(results) == 2 * len(sources)
    assert {r.scenario.id for r in results} == {s.scenario.id for s in sources}
    assert all(r.buyer_agent == "replay:sanity_full_budget_buyer" for r in results)


def test_attacks_expand_with_twins_on_shared_listings():
    cfg = {**BASE, "experiment_name": "x",
           "matrix": {"listings": "all", "seeds": [0], "listing_sample": 1,
                      "sellers": [{"agent": "sanity:accept_above_floor"}],
                      "buyers": [{"agent": "attacks", "categories": ["sob_story"], "twins": True}]}}
    jobs = runner.build_jobs(cfg)
    by_script = {}
    for j in jobs:
        by_script.setdefault(j.buyer["script"], set()).add((j.buyer["twin"], j.listing.id))
    assert jobs and all(len(v) == 2 and len({l for _, l in v}) == 1 for v in by_script.values())
    assert {j.scenario().buyer_persona for j in jobs} == {"attack:sob_story", "twin:sob_story"}


def test_attack_run_scores_effect(tmp_path):
    matrix = {"listings": "all", "seeds": [0], "listing_sample": 1,
              "sellers": [{"agent": "sanity:below_floor"}],
              "buyers": [{"agent": "attacks", "categories": ["extreme_lowball"]}]}
    cfg = write_cfg(tmp_path, "att", matrix=matrix, guardrails={"floor_protection": False})
    rows = [__import__("haggle.metrics", fromlist=["x"]).game_metrics(r) for r in load_results(runner.run(cfg, out_root=tmp_path / "r"))]
    assert {r["attack_kind"] for r in rows} == {"attack", "twin"}
    assert all(r["attack_success"] is not None for r in rows)


def test_guard_and_deadline_specs(tmp_path):
    cfg = {**BASE, "experiment_name": "x",
           "matrix": {"listings": "all", "seeds": [0], "listing_sample": 1,
                      "sellers": [{"agent": "sanity:leaky", "guard": {"leak_filter": True, "grounding": "off"}}],
                      "buyers": [{"agent": "sanity:full_budget", "deadline": "urgent"}]}}
    (job,) = runner.build_jobs(cfg)
    assert job.scenario().buyer_deadline_days == 1.5
    result = runner.play_job(job, cfg, "h", None)
    assert result.seller_agent.endswith("+guarded")
    assert any("redacted" in e or "fixed_on_reask" in e for t in result.turns for e in t.guardrail_events)
