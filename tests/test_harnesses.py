"""Replay, self-play calibration and paired-comparison harness tests (no LLM)."""

import pytest

from haggle.agents.sanity import AcceptAboveFloorSeller, StubbornSeller
from haggle.engine import Game
from haggle.harnesses import ReplayBuyer, paired_difference, replay_divergence, self_play_calibration
from haggle.schemas import Action, GameResult, Listing, Scenario, Turn

LISTING = Listing(id="t1", title="Phone", category="phones", description="", condition_facts=["dent"], asking_price=100)
SC = Scenario(id="t1-s0", listing=LISTING, seller_floor=80, buyer_budget=95, buyer_persona="fair", seed=0)


def source_game(buyer_actions, seller_actions, deal_price=None, outcome="no_deal_turn_limit"):
    turns = []
    for i, (b, s) in enumerate(zip(buyer_actions, seller_actions)):
        turns.append(Turn(index=2 * i, speaker="buyer", action=b))
        if s is not None:
            turns.append(Turn(index=2 * i + 1, speaker="seller", action=s))
    return GameResult(scenario=SC, seller_agent="orig", buyer_agent="llm_buyer:fair", seller_model="m",
                      buyer_model="m", turns=turns, outcome=outcome, deal_price=deal_price, config_hash="h")


def offer(p, msg="hi"):
    return Action(message=msg, action="offer", price=p)


def test_replay_keeps_wait_hours():
    src = source_game([Action(message="hi", action="offer", price=70, wait_hours=12)], [offer(95)])
    g = Game(SC, deadline_days=30)
    g.play(ReplayBuyer(src), StubbornSeller())
    assert g.turns[0].action.wait_hours == 12 and g.current_time_hours == 12


def test_replay_is_verbatim_when_seller_behaves_the_same():
    src = source_game([offer(70), offer(85)], [offer(95), offer(90)])
    buyer = ReplayBuyer(src)
    g = Game(SC)
    g.play(buyer, StubbornSeller())
    buyer_msgs = [t.action for t in g.turns if t.speaker == "buyer"]
    assert buyer_msgs[:2] == [offer(70), offer(85)]
    assert buyer_msgs[2].action == "walk_away"


def test_replay_accept_becomes_offer_if_new_seller_is_pricier():
    accept = Action(message="ok deal", action="accept", price=None)
    src = source_game([offer(70), accept], [offer(88), None], deal_price=88, outcome="deal")
    g = Game(SC)
    g.play(ReplayBuyer(src), StubbornSeller())
    adapted = [t for t in g.turns if t.speaker == "buyer"][1]
    assert adapted.action.action == "offer" and adapted.action.price == 88
    assert adapted.action.message == "ok deal"


def test_replay_accept_kept_when_new_seller_is_cheaper():
    accept = Action(message="ok deal", action="accept", price=None)
    src = source_game([offer(85), accept], [offer(95), None], deal_price=95, outcome="deal")
    g = Game(SC)
    g.play(ReplayBuyer(src), AcceptAboveFloorSeller())
    assert g.deal_price == 85


def test_replay_divergence():
    accept = Action(message="ok", action="accept", price=None)
    src = source_game([offer(70), accept], [offer(88), None], deal_price=88, outcome="deal")
    g = Game(SC)
    buyer, seller = ReplayBuyer(src), StubbornSeller()
    g.play(buyer, seller)
    assert replay_divergence(g.result(buyer, seller)) == pytest.approx(2 / 3)


def test_same_scenario_replayed_across_sellers():
    src = source_game([offer(70), offer(80), offer(90)], [offer(95), offer(92), offer(91)])
    prices = {}
    for seller in (StubbornSeller(), AcceptAboveFloorSeller()):
        g = Game(SC)
        g.play(ReplayBuyer(src), seller)
        prices[seller.name] = g.deal_price
    assert prices["sanity_stubborn_seller"] is None
    assert prices["sanity_accept_above_floor_seller"] == 80


def _row(seller, model_s, model_b, share, sid="x", buyer="b", deal=True):
    return {"seller_agent": seller, "seller_model": model_s, "buyer_model": model_b, "surplus_share": share,
            "deal": deal, "scenario_id": sid, "buyer_agent": buyer}


def test_self_play_calibration_flags_bias():
    rows = [_row("a", "m", "m", s, sid=str(i)) for i, s in enumerate([0.45, 0.5, 0.55] * 10)]
    rows += [_row("a", "biased", "biased", s, sid=str(i)) for i, s in enumerate([0.8, 0.85, 0.9] * 10)]
    rows += [_row("a", "m", "other", 0.0)]
    by_model = {r["model"]: r for r in self_play_calibration(rows)}
    assert by_model["m"]["centred"] and by_model["m"]["games"] == 30
    assert not by_model["biased"]["centred"] and by_model["biased"]["offset_from_half"] == pytest.approx(0.35)


def test_paired_difference_and_win_rate():
    rows = []
    for i in range(10):
        rows += [_row("A", "m", "m", 0.6, sid=str(i)), _row("B", "m", "m", 0.4, sid=str(i))]
    rows += [_row("A", "m", "m", None, sid="nd", deal=False), _row("B", "m", "m", 0.4, sid="nd")]
    out = paired_difference(rows, "A", "B")
    assert out["mean_diff"] == pytest.approx(0.2) and out["pairs"] == 10 and out["dropped"] == 1
    assert out["win_rate_a"] == 1.0
