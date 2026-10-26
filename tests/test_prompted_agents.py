"""Prompt-level switches for Agent A and the LLM buyers (Role 2)."""

import pytest

from fake_llm import FakeClient, action
from haggle.agents.base import FixedAgent
from haggle.agents.buyers import BUYER_DEADLINES, PERSONAS, LLMBuyer
from haggle.agents.prompted_seller import PromptedSeller
from haggle.engine import Game
from haggle.schemas import Action, Listing, Scenario

LISTING = Listing(id="p1", title="iPhone 13", category="phones", description="Used.",
                  condition_facts=["battery 84%", "dent on corner"], asking_price=650)
SCENARIO = Scenario(id="p1-s0", listing=LISTING, seller_floor=527, buyer_budget=613,
                    buyer_persona="test", seed=0)


def seller_prompt(**kw) -> str:
    client = FakeClient([action("hi", "offer", 640)])
    g = Game(SCENARIO)
    g.step(FixedAgent(Action(message="hello", action="offer", price=500)))
    g.step(PromptedSeller(client, **kw))
    return client.calls[0]["messages"][0]["content"]


def buyer_prompt(**kw) -> str:
    client = FakeClient([action("hi", "offer", 500)])
    Game(SCENARIO).step(LLMBuyer(client, **kw))
    return client.calls[0]["messages"][0]["content"]


def test_honesty_instruction_is_a_switch():
    assert "## Honesty" in seller_prompt()
    assert "## Honesty" not in seller_prompt(honesty_instruction=False)


def test_seller_prompt_states_stakes_not_scripted_phases():
    prompt = seller_prompt()
    assert "you get nothing for it" in prompt
    assert "Early on" not in prompt and "As days pass" not in prompt


def test_seller_prompt_has_floor_but_never_budget():
    prompt = seller_prompt()
    assert "527" in prompt and "613" not in prompt


def test_buyer_prompt_has_budget_but_never_floor_or_condition_facts():
    prompt = buyer_prompt()
    assert "613" in prompt and "527" not in prompt
    assert "battery 84%" not in prompt


def test_condition_probe_is_a_switch():
    assert "## Before you agree" in buyer_prompt()
    assert "## Before you agree" not in buyer_prompt(probe_condition=False)


@pytest.mark.parametrize("persona", sorted(PERSONAS))
def test_every_persona_renders(persona):
    assert PERSONAS[persona] in buyer_prompt(persona=persona)


def test_unknown_persona_rejected():
    with pytest.raises(ValueError):
        LLMBuyer(FakeClient(), persona="nope")


def test_deadline_presets_are_ordered():
    assert BUYER_DEADLINES["urgent"] < BUYER_DEADLINES["normal"] < BUYER_DEADLINES["relaxed"]


def test_competing_offer_is_labelled_platform_verified():
    client = FakeClient([action("hi", "offer", 640)])
    g = Game(SCENARIO)
    g.step(FixedAgent(Action(message="hello", action="offer", price=500)))
    seller = PromptedSeller(client)
    obs = g.observation("seller").model_copy(update={"highest_competing_offer": 560, "active_inquirers_count": 2})
    seller.act(obs)
    assert "[Platform-verified: another buyer's current offer is $560" in client.calls[0]["messages"][-1]["content"]


def test_own_past_wait_hours_are_kept_in_history():
    client = FakeClient([action("hi", "offer", 640, wait_hours=12), action("still?", "offer", 630)])
    g = Game(SCENARIO)
    seller = PromptedSeller(client)
    g.step(FixedAgent(Action(message="hello", action="offer", price=500)))
    g.step(seller)  # seller waits 12h before replying
    g.step(FixedAgent(Action(message="?", action="offer", price=510)))
    g.step(seller)
    own_history = client.calls[1]["messages"][2]
    assert own_history["role"] == "assistant" and '"wait_hours":12' in own_history["content"]
