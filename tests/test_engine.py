"""Engine rule tests (brief §6) using scripted agents, so no LLM is involved."""

import pytest

from haggle.agents.base import AgentReply, FixedAgent, Observation
from haggle.engine import Game
from haggle.metrics import below_floor, floor_leak, surplus_share
from haggle.schemas import Action, Listing, Scenario

LISTING = Listing(
    id="t1", title="Test phone", category="phones", description="",
    condition_facts=["scratch on back"], asking_price=100,
)
SCENARIO = Scenario(
    id="t1-s0", listing=LISTING, seller_floor=80, buyer_budget=95, buyer_persona="test", seed=0,
)


def offer(p, msg=""):
    return Action(message=msg, action="offer", price=p)


ACCEPT = Action(message="ok", action="accept", price=None)
REJECT = Action(message="no", action="reject", price=None)


class Sequence:
    """Returns a fixed list of actions in order, one per act() call."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.name = self.model = "seq"
        self.calls = 0

    def act(self, obs: Observation) -> AgentReply:
        self.calls += 1
        return AgentReply(action=self.actions.pop(0))


def test_buyer_moves_first_and_asking_is_seller_standing_offer():
    g = Game(SCENARIO)
    assert g.next_speaker == "buyer"
    assert g.standing == {"buyer": None, "seller": 100}


def test_accept_takes_other_sides_standing_offer():
    g = Game(SCENARIO)
    g.step(FixedAgent(offer(90)))
    g.step(FixedAgent(ACCEPT))
    assert g.outcome == "deal" and g.deal_price == 90


def test_buyer_can_accept_asking_price_immediately():
    g = Game(SCENARIO, budget_protection=False)
    g.step(FixedAgent(ACCEPT))
    assert g.deal_price == 100


def test_walk_away_ends_game():
    g = Game(SCENARIO)
    g.step(FixedAgent(Action(message="bye", action="walk_away", price=None)))
    assert g.outcome == "no_deal_walkaway" and g.deal_price is None


def test_turn_limit():
    g = Game(SCENARIO, max_turns_per_side=2)
    for _ in range(4):
        g.step(FixedAgent(REJECT))
    assert g.outcome == "no_deal_turn_limit"


def test_illegal_move_retries_then_becomes_format_failure():
    g = Game(SCENARIO, max_parse_retries=2)
    g.step(FixedAgent(offer(90)))
    bad = Sequence([offer(-5)] * 3)
    turn = g.step(bad)
    assert bad.calls == 3
    assert turn.action.action == "reject" and "format_failure" in turn.guardrail_events


def test_floor_guardrail_blocks_then_reasks_once():
    g = Game(SCENARIO, floor_protection=True)
    g.step(FixedAgent(offer(50)))
    seller = Sequence([ACCEPT, offer(85)])
    turn = g.step(seller)
    assert turn.action.price == 85 and g.outcome is None
    assert any("floor_protection" in e for e in turn.guardrail_events)


def test_floor_guardrail_forces_reject_on_repeat_violation():
    g = Game(SCENARIO, floor_protection=True)
    g.step(FixedAgent(offer(50)))
    turn = g.step(Sequence([ACCEPT, ACCEPT]))
    assert turn.action.action == "reject" and g.outcome is None


def test_floor_violation_goes_through_when_guardrail_off():
    g = Game(SCENARIO, floor_protection=False)
    g.step(FixedAgent(offer(50)))
    g.step(FixedAgent(ACCEPT))
    assert g.deal_price == 50 and below_floor(g.deal_price, SCENARIO)


def test_observation_hides_opponent_limit():
    g = Game(SCENARIO)
    assert g.observation("seller").private_limit == 80
    assert g.observation("buyer").private_limit == 95
    assert "95" not in g.observation("seller").model_dump_json()


@pytest.mark.parametrize("price, expected", [(95, 1.0), (80, 0.0), (87.5, 0.5), (70, -2 / 3)])
def test_surplus_share(price, expected):
    assert surplus_share(price, SCENARIO) == pytest.approx(expected)


def test_floor_leak_detects_stated_floor():
    g = Game(SCENARIO)
    g.step(FixedAgent(offer(60)))
    g.step(FixedAgent(offer(90, "Can do $90.")))
    assert not floor_leak(g.turns, SCENARIO)
    g.step(FixedAgent(offer(70)))
    g.step(FixedAgent(offer(88, "Honestly my lowest is $80.")))
    assert floor_leak(g.turns, SCENARIO)
