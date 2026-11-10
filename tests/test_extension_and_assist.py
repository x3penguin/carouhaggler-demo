"""Multi-buyer runner mode, RQ6 extension metrics and config safety. No LLM calls."""

import importlib.util
import json

import pytest
import yaml

from haggle import runner
from haggle.agents.base import AgentReply
from haggle.agents.sanity import AcceptAboveFloorSeller
from haggle.config import PROJECT_ROOT, check_models, load_config
from haggle.harnesses import paired_difference
from haggle.metrics import (
    RIVAL_EVENT,
    attack_effect,
    competing_offer_response,
    concession_by_time,
    game_id,
    game_metrics,
    latest_run,
    load_results,
    market_id,
    market_summary,
    seller_moves,
    summarize,
    surplus_shift,
)
from haggle.schemas import Action, GameResult, Listing, Scenario, Turn

LISTING = Listing(id="t1", title="Phone", category="phones", description="", condition_facts=["dent"], asking_price=100)
SC = Scenario(id="t1-s0", listing=LISTING, seller_floor=80, buyer_budget=95, buyer_persona="fair", seed=0)
BASE = {
    "seed": 1, "max_budget_usd": 0.0, "listings_path": "data/listings_sg.json",
    "engine": {"max_turns_per_side": 4, "max_parse_retries": 1},
    "guardrails": {"floor_protection": True}, "bootstrap": {"n_boot": 100, "seed": 0}, "models": {},
}
MARKET = {"seeds": [0], "sellers": [{"agent": "sanity:accept_above_floor"}],
          "markets": {"listing_sample": 3, "deadline_days": 7, "buyer_sets": [[
              {"agent": "sanity:linear", "persona": "a", "deadline": "urgent", "arrival_hours": 0},
              {"agent": "sanity:full_budget", "persona": "b", "deadline": "relaxed", "arrival_hours": 24},
          ]]}}


def game(turns, outcome="no_deal_turn_limit", deal=None, seller="s", persona="fair", sc_id="t1-s0", buyer="b"):
    sc = SC.model_copy(update={"id": sc_id, "buyer_persona": persona})
    return GameResult(scenario=sc, seller_agent=seller, buyer_agent=buyer, seller_model="m", buyer_model="m",
                      turns=turns, outcome=outcome, deal_price=deal, config_hash="h")


def seller_turn(i, action="offer", price=None, hours=0.0, wait=0, events=()):
    return Turn(index=i, speaker="seller", action=Action(message="", action=action, price=price, wait_hours=wait),
                simulated_time_hours=hours, guardrail_events=list(events))


def buyer_turn(i, price=None):
    return Turn(index=i, speaker="buyer", action=Action(message="", action="offer" if price else "chat", price=price))


# ---- multi-buyer runner ------------------------------------------------------------


def test_market_jobs_share_one_floor_with_separate_budgets_and_deadlines():
    jobs = runner.build_jobs({**BASE, "experiment_name": "x", "matrix": MARKET})
    assert len(jobs) == 3
    scenarios = jobs[0].scenarios()
    assert len({s.seller_floor for s in scenarios}) == 1
    assert [s.buyer_deadline_days for s in scenarios] == [1.5, 10.0]
    assert all(s.buyer_budget > s.seller_floor for s in scenarios)
    assert len({s.id for s in scenarios}) == 2


def test_market_run_writes_every_chat_and_sells_once(tmp_path):
    cfg = tmp_path / "m.yaml"
    cfg.write_text(yaml.safe_dump({**BASE, "experiment_name": "m", "matrix": MARKET}))
    out = runner.run(cfg, out_root=tmp_path / "runs")
    results = load_results(out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(results) == 6 and manifest["games_played"] == 6 and manifest["games_planned"] == 6
    by_market = {}
    for r in results:
        by_market.setdefault(market_id(r), []).append(r)
    assert len(by_market) == 3
    for chats in by_market.values():
        assert sum(c.outcome == "deal" for c in chats) <= 1
    summary = market_summary(results)
    assert summary[0]["markets"] == 3 and 0 <= summary[0]["sold"] <= 1


def test_recorder_logs_the_rival_offer_the_seller_actually_saw():
    from haggle.orchestrator import MultiGameOrchestrator
    from haggle.agents.sanity import DelayedBuyer

    seen = []

    class Spy(AcceptAboveFloorSeller):
        def act(self, obs):
            seen.append(obs.highest_competing_offer)
            return AgentReply(action=Action(message="", action="offer", price=obs.listing.asking_price))

    sc1 = SC.model_copy(update={"id": "t1-s0-b1"})
    sc2 = SC.model_copy(update={"id": "t1-s0-b2"})
    orch = MultiGameOrchestrator(LISTING, deadline_days=7, max_turns_per_side=3)
    seller = runner.MarketContextRecorder(Spy())
    orch.add_session("b1", sc1, DelayedBuyer(sc1, wait_hours=4), seller, arrival_hours=0)
    orch.add_session("b2", sc2, DelayedBuyer(sc2, wait_hours=4), seller, arrival_hours=1)
    results = orch.run("h").values()
    logged = [e for r in results for t in r.turns for e in t.guardrail_events if e.startswith(RIVAL_EVENT)]
    assert len(logged) == sum(v is not None for v in seen) and logged


# ---- extension metrics -------------------------------------------------------------


def test_seller_moves_measures_concession_as_share_of_gap():
    r = game([buyer_turn(0, 70), seller_turn(1, price=90), buyer_turn(2, 85),
              seller_turn(3, action="chat"), buyer_turn(4, 85), seller_turn(5, action="accept")])
    moves = seller_moves(r)
    assert [m["concession"] for m in moves] == pytest.approx([0.5, 0.0, 0.25])


def test_concession_by_time_uses_time_left_on_the_seller_deadline():
    late = SC.seller_deadline_days * 24 * 0.9
    r = game([buyer_turn(0, 70), seller_turn(1, price=90), buyer_turn(2), seller_turn(3, price=85, hours=late)])
    bins = {c["time_left"]: c for c in concession_by_time([r])}
    assert bins["early (over 2/3 of deadline left)"]["mean_concession"] == pytest.approx(0.5)
    assert bins["late (under 1/3 left)"]["mean_concession"] == pytest.approx(0.25)


def test_competing_offer_response_keeps_each_condition_and_baseline_apart():
    fake = game([buyer_turn(0, 70), seller_turn(1, price=90)], persona="attack:fake_competing_offer")
    twin = game([buyer_turn(0, 70), seller_turn(1, price=100)], persona="twin:fake_competing_offer")
    market = game([buyer_turn(0, 70), seller_turn(1, price=95, events=[f"{RIVAL_EVENT}88.00"]),
                   buyer_turn(2), seller_turn(3, action="chat")], sc_id="t1-s0-b1")
    out = {c["condition"]: c for c in competing_offer_response([fake, twin], [market])}
    assert out["fake claim"]["mean_concession"] == pytest.approx(0.5)
    assert out["plain twin (baseline)"]["mean_concession"] == 0
    assert out["real rival offer"]["mean_concession"] == pytest.approx(0.25)
    assert out["no rival (baseline)"]["mean_concession"] == 0


def test_rival_event_doesnt_count_as_a_guardrail_block():
    r = game([buyer_turn(0, 70), seller_turn(1, price=90, events=[f"{RIVAL_EVENT}88.00"])])
    row = game_metrics(r)
    assert row["guardrail_blocks"] == 0 and row["seller_redactions"] == 0


# ---- reporting fixes ---------------------------------------------------------------


def test_game_id_separates_buyer_models():
    a = game([]).model_copy(update={"buyer_model": "strong"})
    b = game([]).model_copy(update={"buyer_model": "weak"})
    assert game_id(a) != game_id(b)


def test_crashed_games_are_left_out_of_summaries_but_counted():
    ok = game_metrics(game([buyer_turn(0, 90), seller_turn(1, action="accept")], outcome="deal", deal=90))
    crashed = game_metrics(game([], outcome="error"))
    (s,) = summarize([ok, crashed], group_by=("seller_agent",), n_boot=50)
    assert s["games"] == 1 and s["errors"] == 1 and s["deal_rate"] == 1


def test_attack_effect_is_paired_on_script_and_scenario():
    rows = []
    for i, (a, t) in enumerate([(True, False)] * 5 + [(True, True)] * 5):
        for kind, ok in (("attack", a), ("twin", t)):
            rows.append({"seller_agent": "s", "attack_category": "sob_story", "attack_kind": kind, "attack_success": ok,
                         "scenario_id": f"l{i}", "buyer_agent": f"{kind}:sob_story:sob-{i}", "outcome": "deal"})
    (e,) = attack_effect(rows)
    assert e["pairs"] == 10 and e["effect"] == pytest.approx(0.5)


def test_surplus_shift_pairs_scenarios_and_supports_payoff():
    rows = [{"seller_model": "S", "buyer_model": "W", "surplus_share": 0.8, "seller_payoff": 0.8, "scenario_id": f"l{i}"} for i in range(4)]
    rows += [{"seller_model": "W", "buyer_model": "S", "surplus_share": 0.3, "seller_payoff": 0.0, "scenario_id": f"l{i}"} for i in range(3)]
    out = surplus_shift(rows, "S", "W")
    assert out["pairs"] == 3 and out["shift"] == pytest.approx(0.5)
    assert surplus_shift(rows, "S", "W", metric="seller_payoff")["shift"] == pytest.approx(0.8)


def test_latest_run_never_pools(tmp_path):
    for ts in ("20261101-100000", "20261102-100000"):
        (tmp_path / "pilot" / ts).mkdir(parents=True)
        (tmp_path / "pilot" / ts / "games.jsonl").write_text("")
    assert latest_run(tmp_path / "pilot").name == "20261102-100000"
    assert latest_run(tmp_path / "pilot" / "20261101-100000").name == "20261101-100000"


def test_paired_report_table_finds_pairs_by_seller_variant():
    pytest.importorskip("matplotlib")  # in the optional analysis group
    spec = importlib.util.spec_from_file_location("report", PROJECT_ROOT / "analysis" / "report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    rows = [{"seller_variant": v, "seller_agent": v.split("@")[0], "scenario_id": "a", "buyer_agent": "b", "surplus_share": x}
            for v, x in (("p@m", 0.6), ("q@m", 0.4))]
    table = report.paired_table(rows, [("p@m", "q@m")])
    assert "+0.200" in table
    assert paired_difference(rows, "p", "q")["pairs"] == 1


# ---- configs -----------------------------------------------------------------------


@pytest.mark.parametrize("name,games", [
    ("pilot", 51), ("self_play", 60), ("tournament", 240), ("honesty", 120), ("attacks", 400),
    ("asymmetry", 120), ("opponent", 60),
    ("tournament_finetuned", 120), ("attacks_finetuned", 400),
    ("tournament_base", 240), ("tournament_finetuned_ablation", 120), ("attacks_base", 400),
    ("tournament_dpo", 240), ("attacks_dpo", 400), ("attacks_dpo_v3", 400), ("closing_check", 120),
])
def test_experiment_sizes_match_the_plan(name, games):
    assert len(runner.build_jobs(load_config(f"configs/{name}.yaml"))) == games


def test_extension_is_forty_markets_of_three_buyers_arriving_together():
    jobs = runner.build_jobs(load_config("configs/extension.yaml"))
    assert len(jobs) == 40 and all(len(j.buyers) == 3 for j in jobs)
    assert len({j.listing.id for j in jobs}) == 20
    # Staggered arrivals let the first buyer close every deal before the others arrived (8 Nov run).
    assert all(b.get("arrival_hours", 0) == 0 for j in jobs for b in j.buyers)


def test_solo_baseline_is_buyer_one_of_each_competition_market():
    # RQ6 compares each market with its no-competition twin: same listing, seller, floor, and the
    # same budget and deadline for buyer 1. Only the two rivals differ.
    def first_buyer(name):
        return [(j.seller["agent"], j.scenarios()[0].model_dump_json())
                for j in runner.build_jobs(load_config(f"configs/{name}.yaml"))]
    solo = runner.build_jobs(load_config("configs/extension_solo.yaml"))
    assert all(len(j.buyers) == 1 for j in solo)
    assert first_buyer("extension_solo") == first_buyer("extension")


def test_pilot_covers_every_persona_evenly():
    jobs = runner.build_jobs(load_config("configs/pilot.yaml"))
    assert {p: sum(j.buyer["persona"] == p for j in jobs) for p in ("fair", "tough", "lowballer")} == \
        {"fair": 17, "tough": 17, "lowballer": 17}


def test_honesty_uses_the_same_scenarios_as_the_tournament():
    def scenarios(name):
        return {(j.buyer["persona"], j.scenario().id) for j in runner.build_jobs(load_config(f"configs/{name}.yaml"))}
    assert scenarios("honesty") == scenarios("tournament")


@pytest.mark.parametrize("frontier,other", [("tournament", "tournament_finetuned"), ("attacks", "attacks_finetuned"),
                                            ("tournament", "tournament_base"), ("tournament", "tournament_finetuned_ablation"),
                                            ("attacks", "attacks_base"), ("tournament", "tournament_dpo"),
                                            ("attacks", "attacks_dpo"), ("attacks", "attacks_dpo_v3")])
def test_finetuned_sellers_face_the_same_games_as_the_frontier_sellers(frontier, other):
    # RQ2 compares C with A and D with B game by game, across two separate runs. The whole
    # scenario is compared, so a different floor, budget or listing would fail here.
    def games(name):
        return {(j.buyer.get("script"), j.buyer.get("twin"), j.buyer.get("model"), j.scenario().model_dump_json())
                for j in runner.build_jobs(load_config(f"configs/{name}.yaml"))}
    assert games(other) == games(frontier)


@pytest.mark.parametrize("name,model", [("tournament_finetuned", "finetuned"), ("attacks_finetuned", "finetuned"),
                                        ("tournament_finetuned_ablation", "finetuned"), ("tournament_base", "weak"),
                                        ("attacks_base", "weak"), ("tournament_dpo", "finetuned_dpo"),
                                        ("attacks_dpo", "finetuned_dpo"), ("attacks_dpo_v3", "finetuned_dpo_v3")])
def test_finetuned_configs_only_use_the_finetuned_seller(name, model):
    cfg = load_config(f"configs/{name}.yaml")
    assert {s["model"] for s in cfg["matrix"]["sellers"]} == {model}
    check_models(cfg, {model})  # local models, already set up


def test_every_seller_variant_gets_its_own_name():
    from fake_llm import FakeClient

    for name in ("tournament", "honesty", "attacks", "tournament_finetuned", "attacks_finetuned", "tournament_base",
                 "tournament_finetuned_ablation", "attacks_base", "tournament_dpo", "attacks_dpo", "attacks_dpo_v3"):
        cfg = load_config(f"configs/{name}.yaml")
        job = runner.build_jobs(cfg)[0]
        ctx = runner.BuildContext(cfg, job.scenario(), None)
        ctx.client = lambda key: FakeClient()
        names = [runner.build_agent(runner.SELLERS, s, ctx).name for s in cfg["matrix"]["sellers"]]
        assert len(set(names)) == len(names), names


def test_unset_or_unpriced_hosted_models_are_refused(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    cfg = load_config("configs/pilot.yaml")
    main_seller = cfg["models"]["main_seller"]
    main_seller.update(model="SET_ME_openrouter_model_id", price_in_per_mtok=0.0, price_out_per_mtok=0.0)
    with pytest.raises(ValueError, match="choose a model"):
        check_models(cfg, {"main_seller"})
    main_seller["model"] = "vendor/model"
    with pytest.raises(ValueError, match=r"\$0 price"):
        check_models(cfg, {"main_seller"})
    main_seller["price_in_per_mtok"] = 1.0
    check_models(cfg, {"main_seller"})
    check_models(cfg, {"weak"})


# ---- report helpers for RQ2, RQ6 and the judge check -------------------------------


def _report():
    pytest.importorskip("matplotlib")  # in the optional analysis group
    spec = importlib.util.spec_from_file_location("report", PROJECT_ROOT / "analysis" / "report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    return report


def test_competition_table_pairs_each_market_with_its_one_buyer_twin():
    report = _report()
    sale = lambda hours: [seller_turn(0, action="accept", hours=hours)]
    solo = [game(sale(20), outcome="deal", deal=85, sc_id="t1-s0-b1")]  # payoff (85 - 80) / (100 - 80) = 0.25
    comp = [game([], outcome="no_deal_item_sold", sc_id="t1-s0-b1"),
            game(sale(8), outcome="deal", deal=90, sc_id="t1-s0-b2", persona="tough")]  # payoff 0.5
    comp = [g.model_copy(update={"config_hash": "other"}) for g in comp]  # another config must still pair
    (one,), (three,) = report.market_rows(solo, "solo"), report.market_rows(comp, "three buyers")
    assert one["market"] == three["market"] == "t1-s0" and three["winner"] == "tough"
    assert "+0.25" in report.competition_table(solo, comp)


def test_judge_agreement_combines_runs_and_has_a_major_only_check(tmp_path):
    from haggle.judge import label_id
    report = _report()
    for run, gid, major in (("a", "g1", 0), ("b", "g2", 1)):
        (tmp_path / run).mkdir()
        rec = {"game_id": gid, "violations": 1, "major_violations": major, "judge_floor_leak": False, "error": None}
        (tmp_path / run / "honesty.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    labels = tmp_path / "labels.csv"
    labels.write_text("label_id,listing,human_dishonest,human_floor_leak,notes\n"
                      f"{label_id('g1')},Phone,0,0,minor flag only\n{label_id('g2')},Phone,1,0,\n", encoding="utf-8")
    any_v = report.judge_agreement(labels, tmp_path / "a", tmp_path / "b")["dishonest"]
    major = report.judge_agreement(labels, tmp_path / "a", tmp_path / "b", major_only=True)["dishonest"]
    assert (any_v["n"], any_v["tp"], any_v["fp"]) == (2, 1, 1)
    assert (major["tp"], major["fp"], major["tn"]) == (1, 0, 1)
    assert "minor flag only" in report.judge_disagreements(labels, tmp_path / "a", tmp_path / "b")


def test_chart_labels_use_design_letters_and_drop_the_provider():
    report = _report()
    s = {"seller_agent": "agentic_seller[tools+plan+reflect]", "seller_model": "provider/some-model"}
    assert report.short_label(s, ("seller_agent", "seller_model")) == "B · some-model"
