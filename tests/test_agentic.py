"""Agent B and the agentic buyer: stage switches, tools, memory isolation, retries (Role 2)."""

import pytest

from fake_llm import RouterClient, action
from haggle.agents.agentic import AgenticBuyer, AgenticSeller
from haggle.agents.base import FixedAgent
from haggle.engine import Game
from haggle.orchestrator import MultiGameOrchestrator
from haggle.schemas import Action, Listing, Scenario

LISTING = Listing(id="phone-001", title="iPhone 13, 128GB, Midnight", category="phones",
                  description="Used.", condition_facts=["battery 84%", "dent on corner"], asking_price=620)
SCENARIO = Scenario(id="s", listing=LISTING, seller_floor=500, buyer_budget=580, buyer_persona="t", seed=0)

PLAN = {"estimate_of_their_limit": 560, "strategy": "Small concession.", "target_price": 600,
        "next_move": "offer 600 to keep momentum"}
LESSON = {"lesson": "Buyer moves in $20 steps; hold firm."}


def routes(**over):
    base = {"Plan": PLAN, "Reflection": LESSON, "Action": action("How about $600?", "offer", 600)}
    base.update(over)
    return base


def play_two_seller_turns(seller):
    g = Game(SCENARIO)
    g.step(FixedAgent(Action(message="hi, $450?", action="offer", price=450)))
    g.step(seller)
    g.step(FixedAgent(Action(message="$470?", action="offer", price=470)))
    g.step(seller)
    return g


@pytest.mark.parametrize("tools, plan, reflect, expected", [
    (False, False, False, ["Action", "Action"]),                          # identical to Agent A
    (True, False, False, ["Action", "Action"]),                           # tools need no LLM call
    (False, True, False, ["Plan", "Action", "Plan", "Action"]),
    (False, False, True, ["Action", "Reflection", "Action"]),             # nothing to reflect on in turn 1
    (True, True, True, ["Plan", "Action", "Reflection", "Plan", "Action"]),
])
def test_stage_switches_control_llm_calls(tools, plan, reflect, expected):
    client = RouterClient(routes())
    play_two_seller_turns(AgenticSeller(client, use_tools=tools, use_planning=plan, use_reflection=reflect))
    assert client.schemas() == expected


def test_variant_names_identify_the_ablation():
    c = RouterClient(routes())
    assert AgenticSeller(c).name == "agentic_seller[tools+plan+reflect]"
    assert AgenticSeller(c, use_planning=False, use_reflection=False).name == "agentic_seller[tools]"
    assert AgenticSeller(c, use_tools=False, use_planning=False, use_reflection=False).name == "agentic_seller[none]"


def test_tools_are_logged_and_shown_privately():
    client = RouterClient(routes())
    g = play_two_seller_turns(AgenticSeller(client))
    first, second = g.turns[1], g.turns[3]
    assert [c["name"] for c in first.tool_calls] == ["price_lookup", "walk_away_value", "calculator"]
    assert [c["name"] for c in second.tool_calls] == ["walk_away_value", "calculator"]  # lookup cached
    act_prompt = client.calls[1]["messages"][-1]["content"]
    assert "[Tool] Market check" in act_prompt and "[Your private plan]" in act_prompt
    assert "Market check" in first.private_notes


def test_private_notes_never_reach_the_opponent():
    client = RouterClient(routes())
    g = play_two_seller_turns(AgenticSeller(client))
    buyer_view = g.observation("buyer").model_dump_json()
    assert "private plan" not in buyer_view and "Market check" not in buyer_view
    assert "Buyer moves in $20 steps" not in buyer_view


def test_reflection_lessons_carry_into_the_next_plan():
    client = RouterClient(routes())
    play_two_seller_turns(AgenticSeller(client))
    second_plan_prompt = client.calls[3]["messages"][-1]["content"]
    assert client.calls[3]["schema"] == "Plan"
    assert "Buyer moves in $20 steps" in second_plan_prompt


def test_retry_reuses_the_plan_instead_of_replanning():
    # First action is unparseable, so the engine retries the same turn with feedback.
    client = RouterClient(routes(Action=["not json", action("How about $600?", "offer", 600)]))
    g = Game(SCENARIO)
    g.step(FixedAgent(Action(message="$450?", action="offer", price=450)))
    turn = g.step(AgenticSeller(client))
    assert client.schemas() == ["Plan", "Action", "Action"]
    assert turn.parse_retries == 1 and turn.action.price == 600


def test_new_game_starts_with_fresh_memory():
    client = RouterClient(routes())
    seller = AgenticSeller(client)
    play_two_seller_turns(seller)
    client.calls.clear()
    g = Game(SCENARIO)
    g.step(FixedAgent(Action(message="hello again", action="offer", price=450)))
    g.step(seller)
    assert "Buyer moves in $20 steps" not in client.calls[0]["messages"][-1]["content"]
    assert client.schemas()[0] == "Plan"  # no reflection on the first turn of a new game


def test_memory_is_isolated_per_chat_in_multi_buyer_mode():
    def lesson_for_this_chat(messages, schema):
        sid = "A" if "A offers" in messages[-1]["content"] else "B"
        return {"lesson": f"Chat {sid} buyer noted."}

    client = RouterClient(routes(Reflection=lesson_for_this_chat))
    seller = AgenticSeller(client, use_tools=False)
    orch = MultiGameOrchestrator(LISTING, enforce_floor_protection=True, max_turns_per_side=3)
    for sid in ("A", "B"):
        sc = SCENARIO.model_copy(update={"id": sid})
        buyer = FixedAgent(Action(message=f"{sid} offers", action="offer", price=450))
        orch.add_session(sid, sc, buyer, seller, arrival_hours=0 if sid == "A" else 0.5)
    for _ in range(8):
        orch.step()
    plan_prompts = [c["messages"][-1]["content"] for c in client.calls if c["schema"] == "Plan"]
    # Chat B's plans must never mention chat A's lesson, and vice versa.
    for prompt in plan_prompts:
        assert not ("Chat A buyer" in prompt and "Chat B buyer" in prompt)
    assert seller._memories[("A", LISTING.id)].lessons != seller._memories[("B", LISTING.id)].lessons


def test_bad_plan_json_falls_back_to_acting_without_a_plan():
    client = RouterClient(routes(Plan="garbage"))
    g = Game(SCENARIO)
    g.step(FixedAgent(Action(message="$450?", action="offer", price=450)))
    turn = g.step(AgenticSeller(client))
    assert turn.action.price == 600 and "[Your private plan]" not in (turn.private_notes or "")


def test_agentic_buyer_uses_market_check_and_buyer_prompt():
    client = RouterClient(routes(Action=action("Can do $480?", "offer", 480)))
    buyer = AgenticBuyer(client, persona="tough")
    g = Game(SCENARIO)
    turn = g.step(buyer)
    assert buyer.name == "agentic_buyer:tough[tools+plan+reflect]"
    assert [c["name"] for c in turn.tool_calls] == ["price_lookup"]
    plan_prompt = client.calls[0]["messages"][-1]["content"]
    assert "seller's minimum acceptable price" in plan_prompt
    assert "500" not in client.calls[0]["messages"][0]["content"]  # never sees the seller's floor
