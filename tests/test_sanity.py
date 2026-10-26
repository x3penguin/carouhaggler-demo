"""Run full games with rule-based agents and check metrics return the known values."""

import pytest

from haggle.agents.sanity import (
    AcceptAboveFloorSeller,
    BackwardOfferAgent,
    BelowFloorSeller,
    FloorBuyer,
    FullBudgetBuyer,
    LeakySeller,
    LinearConcessionBuyer,
    StubbornSeller,
)
from haggle.engine import Game
from haggle.metrics import below_floor, floor_leak, format_failures, surplus_share
from haggle.scenarios import load_listings, make_scenario
from haggle.schemas import Listing, Scenario

LISTINGS = load_listings("data/listings_sg.json")
SCENARIOS = [make_scenario(lst, seed) for lst in LISTINGS for seed in range(5)]
# Scenarios where the asking price is above budget, so a stubborn seller can never close.
PRICEY = [s for s in SCENARIOS if s.buyer_budget < s.listing.asking_price]


def play(scenario, buyer, seller, **kw):
    g = Game(scenario, **kw)
    g.play(buyer, seller)
    return g


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda s: s.id)
def test_full_budget_buyer_gives_surplus_share_one(sc):
    g = play(sc, FullBudgetBuyer(sc), AcceptAboveFloorSeller())
    assert g.outcome == "deal" and g.deal_price == sc.buyer_budget
    assert surplus_share(g.deal_price, sc) == pytest.approx(1.0)


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda s: s.id)
def test_floor_buyer_gives_surplus_share_zero(sc):
    g = play(sc, FloorBuyer(sc), AcceptAboveFloorSeller())
    assert g.outcome == "deal" and g.deal_price == sc.seller_floor
    assert surplus_share(g.deal_price, sc) == pytest.approx(0.0)
    assert not below_floor(g.deal_price, sc)


@pytest.mark.parametrize("sc", PRICEY, ids=lambda s: s.id)
def test_stubborn_seller_never_closes_against_linear_buyer(sc):
    g = play(sc, LinearConcessionBuyer(sc), StubbornSeller())
    assert g.outcome == "no_deal_turn_limit" and g.deal_price is None


def test_stubborn_seller_deal_rate_is_zero_across_scenarios():
    deals = sum(play(sc, LinearConcessionBuyer(sc), StubbornSeller()).outcome == "deal" for sc in PRICEY)
    assert PRICEY and deals == 0


def test_linear_buyer_has_predictable_turn_count():
    listing = Listing(id="t", title="t", category="c", description="", condition_facts=["x"], asking_price=100)
    sc = Scenario(id="t-s0", listing=listing, seller_floor=80, buyer_budget=95, buyer_persona="t", seed=0)
    buyer = LinearConcessionBuyer(sc, steps=5, start_frac=0.6)
    g = play(sc, buyer, AcceptAboveFloorSeller())
    # offers 60, 68.75, 77.5, 86.25: first one >= floor 80 is the 4th buyer turn (index 6); seller accepts on index 7.
    assert g.outcome == "deal" and g.deal_price == pytest.approx(86.25)
    assert len(g.turns) == 8


def test_leaky_seller_is_flagged_and_honest_seller_is_not():
    sc = PRICEY[0]
    leaky = play(sc, LinearConcessionBuyer(sc), LeakySeller())
    quiet = play(sc, LinearConcessionBuyer(sc), StubbornSeller())
    assert floor_leak(leaky.turns, sc)
    assert not floor_leak(quiet.turns, sc)


@pytest.mark.parametrize("sc", [s for s in SCENARIOS if s.seller_floor > 1][:10], ids=lambda s: s.id)
def test_floor_guardrail_blocks_below_floor_seller(sc):
    low = FloorBuyer(sc)
    low.floor = sc.seller_floor - 10  # offer below floor, to tempt the seller
    on = play(sc, low, BelowFloorSeller(), floor_protection=True)
    assert not below_floor(on.deal_price, sc)
    assert any("floor_protection" in e for t in on.turns for e in t.guardrail_events)

    off = play(sc, low, BelowFloorSeller(), floor_protection=False)
    assert off.outcome == "deal" and below_floor(off.deal_price, sc)


def test_no_format_failures_from_sanity_agents():
    sc = SCENARIOS[0]
    g = play(sc, FullBudgetBuyer(sc), AcceptAboveFloorSeller())
    assert format_failures(g.turns, "buyer") == 0 and format_failures(g.turns, "seller") == 0


def test_backward_offer_agent_moves_backward():
    sc = PRICEY[0]
    g = play(sc, BackwardOfferAgent(), StubbornSeller(), budget_protection=False, max_turns_per_side=3)
    prices = [t.action.price for t in g.turns if t.speaker == "buyer"]
    assert prices == [90, 80, 85]
    assert any(b > a for a, b in zip(prices, prices[1:])) and any(b < a for a, b in zip(prices, prices[1:]))
