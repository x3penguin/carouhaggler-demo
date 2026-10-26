"""Attack scripts, matched twins and the attacker agent (RQ3, Role 2)."""

from collections import Counter

import pytest

from fake_llm import FakeClient
from haggle.agents.attackers import SUCCESS, TACTICS, AttackBuyer, AttackScript, load_attacks
from haggle.agents.base import FixedAgent
from haggle.agents.sanity import StubbornSeller
from haggle.engine import Game
from haggle.schemas import Action, Listing, Scenario

ATTACKS = load_attacks()
LISTING = Listing(id="p1", title="iPhone 13", category="phones", description="Need to clear before moving.",
                  condition_facts=["dent"], asking_price=600)
SCENARIO = Scenario(id="s", listing=LISTING, seller_floor=480, buyer_budget=560, buyer_persona="attack", seed=0)


# ---- the data file -----------------------------------------------------------------


def test_about_fifty_scripts_with_unique_ids():
    assert len(ATTACKS) >= 49
    assert len({a.id for a in ATTACKS}) == len(ATTACKS)


def test_every_category_is_covered_and_has_a_tactic_and_success_rule():
    counts = Counter(a.category for a in ATTACKS)
    assert set(counts) == set(SUCCESS) == set(TACTICS)
    assert min(counts.values()) >= 6


@pytest.mark.parametrize("script", ATTACKS, ids=lambda a: a.id)
def test_script_is_well_formed(script: AttackScript):
    assert 2 <= len(script.steps) <= 5
    for step in script.steps:
        has_price = step.price_frac is not None
        assert ("{p}" in step.message) == has_price or script.category == "prompt_injection"
        if has_price:
            assert 0.2 <= step.price_frac < 0.7  # aimed below a typical floor (70-90% of asking)


def test_success_rules_match_the_brief():
    assert SUCCESS["floor_extraction"] == "floor_leak"
    assert SUCCESS["prompt_injection"] == "either"
    assert SUCCESS["sob_story"] == "below_floor"


# ---- the agent -----------------------------------------------------------------------


def play(buyer, seller=None, turns=10):
    g = Game(SCENARIO, budget_protection=False, floor_protection=False)
    seller = seller or StubbornSeller()
    while not g.is_over and len(g.turns) < turns:
        g.step(buyer if g.next_speaker == "buyer" else seller)
    return g


def buyer_turns(g):
    return [t.action for t in g.turns if t.speaker == "buyer"]


@pytest.mark.parametrize("script", ATTACKS, ids=lambda a: a.id)
def test_twin_follows_the_same_price_path_and_timing(script):
    attack = buyer_turns(play(AttackBuyer(script)))
    twin = buyer_turns(play(AttackBuyer(script, twin=True)))
    assert [(a.action, a.price, a.wait_hours) for a in attack] == [(t.action, t.price, t.wait_hours) for t in twin]
    assert all(a.message != t.message for a, t in zip(attack, twin) if a.action != "walk_away")


def test_prices_follow_the_script_then_walk_away():
    script = next(a for a in ATTACKS if a.id == "sob-01")
    actions = buyer_turns(play(AttackBuyer(script)))
    assert [a.price for a in actions[:3]] == [300, 312, 330]  # 0.50, 0.52, 0.55 of $600
    assert actions[-1].action == "walk_away"


def test_wait_hours_from_script_are_used():
    script = next(a for a in ATTACKS if a.id == "ddl-02")
    assert [a.wait_hours for a in buyer_turns(play(AttackBuyer(script)))][:3] == [0, 48, 48]


def test_accepts_when_seller_comes_down_to_our_offer():
    script = next(a for a in ATTACKS if a.id == "low-01")
    seller = FixedAgent(Action(message="ok fine", action="offer", price=150))  # below our first offer of 180
    g = play(AttackBuyer(script), seller)
    assert g.outcome == "deal" and g.deal_price == 150


def test_twin_text_contains_no_tactic():
    script = next(a for a in ATTACKS if a.id == "inj-01")
    texts = " ".join(a.message for a in buyer_turns(play(AttackBuyer(script, twin=True))))
    assert "SYSTEM" not in texts and "override" not in texts.lower()


def test_persona_labels():
    script = ATTACKS[0]
    assert AttackBuyer(script).persona == f"attack:{script.category}"
    assert AttackBuyer(script, twin=True).persona == f"twin:{script.category}"


def test_llm_mode_rewrites_words_but_code_keeps_price():
    script = next(a for a in ATTACKS if a.id == "sob-01")
    client = FakeClient([{"message": "My phone broke before exams, could you do $300?"},
                         {"message": "Please, $312 is all I have."}])
    buyer = AttackBuyer(script, client=client)
    g = Game(SCENARIO, floor_protection=False)
    g.step(buyer)
    g.step(StubbornSeller())
    g.step(buyer)
    actions = buyer_turns(g)
    assert actions[0].message.startswith("My phone broke") and actions[0].price == 300
    assert actions[1].price == 312
    assert buyer.name.endswith("+llm")
    prompt = client.calls[0]["messages"]
    assert "hardship" in prompt[0]["content"] and "Your price this turn is $300" in prompt[-1]["content"]
    assert "480" not in prompt[0]["content"]  # attacker never sees the floor


def test_llm_mode_falls_back_to_script_on_bad_output():
    script = next(a for a in ATTACKS if a.id == "urg-01")
    g = Game(SCENARIO)
    g.step(AttackBuyer(script, client=FakeClient(["not json"])))
    assert g.turns[0].action.message == script.steps[0].message.replace("{p}", "360")
