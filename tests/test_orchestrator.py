"""Tests for MultiGameOrchestrator: simulated time and concurrent buyer competition."""

import pytest

from haggle.agents.sanity import (
    AcceptAboveFloorSeller,
    CompetitiveBuyer,
    DelayedBuyer,
    FloorBuyer,
    FullBudgetBuyer,
    StubbornSeller,
)
from haggle.orchestrator import MultiGameOrchestrator
from haggle.schemas import Action, Listing, Scenario

LISTING = Listing(
    id="phone-t1",
    title="iPhone 13 128GB Midnight",
    category="phones",
    description="Test item",
    condition_facts=["scratch on back"],
    asking_price=600,
)


def make_test_scenario(session_id: str, floor: float = 480, budget: float = 550) -> Scenario:
    return Scenario(
        id=f"sc-{session_id}",
        listing=LISTING,
        seller_floor=floor,
        buyer_budget=budget,
        buyer_persona="test",
        seed=1,
    )


def test_orchestrator_initializes_and_schedules_first_event():
    orch = MultiGameOrchestrator(LISTING, deadline_days=7.0)
    sc = make_test_scenario("b1")
    orch.add_session("b1", sc, FullBudgetBuyer(sc), AcceptAboveFloorSeller(), arrival_hours=0.0)

    assert len(orch.event_queue) == 1
    assert orch.simulated_time_hours == 0.0
    assert orch.inventory_status == "available"


def test_deal_in_one_session_locks_inventory_and_closes_other():
    orch = MultiGameOrchestrator(LISTING, deadline_days=7.0)
    sc1 = make_test_scenario("b1", budget=520)
    sc2 = make_test_scenario("b2", budget=550)

    # Buyer 1 arrives Day 0 (hour 0) with a 24h delay between moves
    orch.add_session("b1", sc1, DelayedBuyer(sc1, wait_hours=24), AcceptAboveFloorSeller(), arrival_hours=0.0)
    # Buyer 2 arrives Day 0.5 (hour 12) and offers full budget immediately with 0 delay
    orch.add_session("b2", sc2, FullBudgetBuyer(sc2), AcceptAboveFloorSeller(), arrival_hours=12.0)

    results = orch.run()

    # Buyer 2 should have closed the deal at hour 12
    assert orch.inventory_status == "sold"
    assert orch.sold_to == "b2"
    assert orch.deal_price == 550

    # Session 2 has a deal
    assert results["b2"].outcome == "deal"
    assert results["b2"].deal_price == 550

    # Session 1 was cut short with item sold
    assert results["b1"].outcome == "no_deal_item_sold"
    assert results["b1"].deal_price is None


def test_highest_competing_offer_tracked_across_sessions():
    orch = MultiGameOrchestrator(LISTING, deadline_days=7.0)
    sc1 = make_test_scenario("b1", budget=500)
    sc2 = make_test_scenario("b2", budget=560)

    # Buyer 1 arrives at hour 0.0 and delays 4 hours before next turn
    orch.add_session("b1", sc1, DelayedBuyer(sc1, wait_hours=4), StubbornSeller(), arrival_hours=0.0)
    # Buyer 2 arrives at hour 2.0
    orch.add_session("b2", sc2, FullBudgetBuyer(sc2), StubbornSeller(), arrival_hours=2.0)

    # Step 1: b1 steps at hour 0.0 -> offers 500, next turn scheduled at 4.0
    orch.step()
    assert orch.sessions["b1"].game.standing["buyer"] == 500
    assert orch.highest_competing_offer(exclude_session_id="b2") == 500

    # Step 2: earliest now is b2 at hour 2.0 -> offers 560
    orch.step()
    assert orch.sessions["b2"].game.standing["buyer"] == 560
    assert orch.highest_competing_offer(exclude_session_id="b1") == 560


def test_deadline_terminates_all_unclosed_sessions():
    # 2 days deadline = 48 hours
    orch = MultiGameOrchestrator(LISTING, deadline_days=2.0)
    sc = make_test_scenario("b1")

    # Buyer that waits 30 hours per move against a stubborn seller
    orch.add_session("b1", sc, DelayedBuyer(sc, wait_hours=30), StubbornSeller(), arrival_hours=0.0)

    results = orch.run()

    assert orch.inventory_status == "expired"
    assert results["b1"].outcome == "no_deal_deadline"
    assert orch.simulated_time_hours > 48.0


def test_competitive_buyer_outbids_competing_offer():
    orch = MultiGameOrchestrator(LISTING, deadline_days=7.0)
    sc1 = make_test_scenario("b1", budget=500)
    sc2 = make_test_scenario("b2", budget=540)

    # Buyer 1 offers 480 and waits 4h
    orch.add_session("b1", sc1, DelayedBuyer(sc1, wait_hours=4), StubbornSeller(), arrival_hours=0.0)
    # Buyer 2 arrives at hour 2.0 and outbids rival by 15
    orch.add_session("b2", sc2, CompetitiveBuyer(sc2, increment=15.0), StubbornSeller(), arrival_hours=2.0)

    orch.step()  # b1 steps at hour 0 -> offers 500 (budget), next at 4h
    orch.step()  # b2 steps at hour 2 -> sees competing 500 -> offers 500 + 15 = 515
    assert orch.sessions["b2"].game.standing["buyer"] == 515
