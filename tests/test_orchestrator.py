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

    # Hour 0: b1 decides to offer 500 after a 4h wait, so nothing is visible yet.
    orch.step()
    assert orch.sessions["b1"].game.standing["buyer"] is None
    # Hour 2: b2 arrives and offers 560 at once.
    orch.step()
    assert orch.sessions["b2"].game.standing["buyer"] == 560
    assert orch.highest_competing_offer(exclude_session_id="b2") is None
    # Hour 4: b1's offer is delivered and becomes a competing offer for b2's chat.
    while orch.simulated_time_hours < 4.0:
        orch.step()
    assert orch.sessions["b1"].game.standing["buyer"] == 500
    assert orch.highest_competing_offer(exclude_session_id="b2") == 500


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
    # Buyer 2 arrives at hour 5, after b1's offer (sent with a 4h wait) has landed
    orch.add_session("b2", sc2, CompetitiveBuyer(sc2, increment=15.0), StubbornSeller(), arrival_hours=5.0)

    while orch.sessions["b2"].game.standing["buyer"] is None:
        orch.step()
    assert orch.sessions["b2"].game.standing["buyer"] == 515


# ---- regression: retries and guardrail re-asks in multi-buyer mode -------------


class _ReaskAwareSeller:
    """Tries to accept below floor; counters at 500 once the engine sends feedback."""

    name = model = "reask_aware"

    def __init__(self):
        self.calls = 0
        self.seen = []

    def act(self, obs):
        from haggle.agents.base import AgentReply

        self.calls += 1
        self.seen.append(obs)
        if obs.feedback is None:
            return AgentReply(action=Action(message="ok", action="accept"), tokens_in=100)
        return AgentReply(action=Action(message="500?", action="offer", price=500), tokens_in=100)


def test_guardrail_reask_calls_agent_again_with_feedback():
    seller = _ReaskAwareSeller()
    orch = MultiGameOrchestrator(LISTING, enforce_floor_protection=True)
    sc = make_test_scenario("b1", floor=480)
    orch.add_session("b1", sc, FloorBuyer(make_test_scenario("b1", floor=400)), seller)
    orch.step()  # buyer offers 400 (below the real floor of 480)
    orch.step()  # seller tries to accept, gets blocked, is asked again
    turn = orch.sessions["b1"].game.turns[-1]
    assert seller.calls == 2
    assert turn.action.action == "offer" and turn.action.price == 500
    assert "guardrail_forced_reject" not in turn.guardrail_events


def test_tokens_counted_once_per_model_call():
    seller = _ReaskAwareSeller()
    orch = MultiGameOrchestrator(LISTING, enforce_floor_protection=True)
    orch.add_session("b1", make_test_scenario("b1"), FloorBuyer(make_test_scenario("b1", floor=400)), seller)
    orch.step()
    orch.step()
    assert orch.sessions["b1"].game.turns[-1].tokens_in == 100 * seller.calls


def test_market_context_and_session_id_reach_every_attempt():
    seller = _ReaskAwareSeller()
    orch = MultiGameOrchestrator(LISTING, enforce_floor_protection=True)
    orch.add_session("chat-7", make_test_scenario("chat-7"), FloorBuyer(make_test_scenario("x", floor=400)), seller)
    orch.step()
    orch.step()
    assert [o.session_id for o in seller.seen] == ["chat-7", "chat-7"]
    assert all(o.deadline_hours == orch.deadline_hours for o in seller.seen)
    assert seller.seen[1].feedback is not None


# ---- regression: timing and visibility bugs (review of 31 Oct) -------------------


class _Script:
    def __init__(self, actions, name="script"):
        self.actions, self.name, self.model, self.seen = list(actions), name, "m", []

    def act(self, obs):
        from haggle.agents.base import AgentReply

        self.seen.append(obs)
        return AgentReply(action=self.actions.pop(0) if self.actions else Action(message="", action="chat"))


def _offer(p, wait=0):
    return Action(message="", action="offer", price=p, wait_hours=wait)


def test_late_buyer_turns_use_the_shared_clock():
    orch = MultiGameOrchestrator(LISTING, deadline_days=7.0)
    orch.add_session("b1", make_test_scenario("b1"), _Script([_offer(500)]), _Script([_offer(590)]), arrival_hours=48.0)
    orch.step()
    assert orch.sessions["b1"].game.turns[0].simulated_time_hours == 48.0


def test_buyer_keeps_its_own_deadline():
    buyer = _Script([_offer(500)])
    sc = make_test_scenario("b1").model_copy(update={"buyer_deadline_days": 1.5})
    orch = MultiGameOrchestrator(LISTING, deadline_days=7.0)
    orch.add_session("b1", sc, buyer, _Script([]), arrival_hours=10.0)
    orch.step()
    assert buyer.seen[0].deadline_hours == 10.0 + 36.0


def test_finished_and_future_chats_do_not_count_as_competition():
    walk = Action(message="bye", action="walk_away")
    seller = _Script([_offer(590)] * 5)
    orch = MultiGameOrchestrator(LISTING, deadline_days=7.0)
    orch.add_session("gone", make_test_scenario("gone"), _Script([_offer(540), walk]), _Script([_offer(590)]))
    orch.add_session("now", make_test_scenario("now"), _Script([_offer(500)]), seller, arrival_hours=10.0)
    orch.add_session("later", make_test_scenario("later"), _Script([]), _Script([]), arrival_hours=100.0)
    while not seller.seen:
        orch.step()
    assert orch.sessions["gone"].game.outcome == "no_deal_walkaway"
    assert seller.seen[0].highest_competing_offer is None
    assert seller.seen[0].active_inquirers_count == 1


def test_delayed_accept_lands_after_its_wait():
    orch = MultiGameOrchestrator(LISTING, deadline_days=7.0)
    sc1, sc2 = make_test_scenario("b1", budget=600), make_test_scenario("b2", budget=600)
    orch.add_session("slow", sc1, _Script([Action(message="ok", action="accept", wait_hours=24)]), _Script([]))
    orch.add_session("fast", sc2, _Script([Action(message="ok", action="accept")]), _Script([]), arrival_hours=2.0)
    results = orch.run()
    assert orch.sold_to == "fast" and results["slow"].outcome == "no_deal_item_sold"


def test_buyers_can_be_blinded_to_rival_offers():
    b2 = _Script([_offer(500)])
    orch = MultiGameOrchestrator(LISTING, deadline_days=7.0, buyers_see_competing_offers=False)
    orch.add_session("b1", make_test_scenario("b1"), _Script([_offer(540)]), _Script([_offer(590)]))
    orch.add_session("b2", make_test_scenario("b2"), b2, _Script([]), arrival_hours=1.0)
    while not b2.seen:
        orch.step()
    assert b2.seen[0].highest_competing_offer is None


def test_one_floor_per_listing():
    orch = MultiGameOrchestrator(LISTING)
    orch.add_session("a", make_test_scenario("a", floor=480), _Script([]), _Script([]))
    with pytest.raises(ValueError, match="floor"):
        orch.add_session("b", make_test_scenario("b", floor=300), _Script([]), _Script([]))


def test_notes_are_not_errors_and_config_hash_is_kept():
    orch = MultiGameOrchestrator(LISTING)
    orch.add_session("a", make_test_scenario("a", budget=600), _Script([Action(message="ok", action="accept")]), _Script([]))
    orch.add_session("b", make_test_scenario("b"), _Script([_offer(500)]), _Script([]), arrival_hours=5.0)
    results = orch.run(config_hash="abc")
    assert results["b"].outcome == "no_deal_item_sold" and results["b"].error is None
    assert results["a"].config_hash == "abc"
