"""CraigslistBargain conversion (RQ2). Uses small hand-written dialogues, so no download is needed."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # finetune/ isn't part of the package

from finetune.prepare_data import BUYER_OPENER, Skip, build_split, convert, scrub
from haggle.agents.prompted_seller import PromptedSeller
from haggle.schemas import Action

SELLER, BUYER = 1, 0


def msg(agent, text, intent="unknown", price=None):
    return {"agent": agent, "action": "message", "data": text, "metadata": {"intent": intent, "price": price}}


def offer(agent, price):
    return {"agent": agent, "action": "offer", "data": {"price": price, "sides": ""}, "metadata": {"intent": "offer", "price": price}}


def formal(agent, action):
    return {"agent": agent, "action": action, "data": None, "metadata": {"intent": action}}


def dialogue(events, deal=None, category="phone", asking=100, buyer_target=60, uuid="C_1"):
    item = {"Category": category, "Price": asking, "Title": "Phone", "Description": ["A phone."], "Images": []}
    return {
        "uuid": uuid,
        "scenario": {"category": category, "kbs": [
            {"personal": {"Role": "buyer", "Target": buyer_target, "Bottomline": None}, "item": item},
            {"personal": {"Role": "seller", "Target": asking, "Bottomline": None}, "item": item},
        ]},
        "outcome": {"reward": 1, "offer": {"price": deal, "sides": ""}} if deal else {"reward": 0, "offer": None},
        "events": events,
    }


# A good deal: the $90 close is 75% of the way from the buyer's $60 target to the $100 ask.
GOOD_DEAL = dialogue([
    msg(BUYER, "Is it scratched?", "inquiry"),
    msg(SELLER, "Barely, it's in great shape.", "inform"),
    msg(BUYER, "Would you take $70?", "init-price", 70),
    msg(SELLER, "I could do $92.", "counter-price", 92),
    msg(BUYER, "Ok, $90 then.", "counter-price", 90),
    formal(SELLER, "accept"),
], deal=90)


def targets(examples):
    return [Action.model_validate_json(e[-1]["content"]) for e in examples]


def test_seller_moves_map_onto_our_actions():
    assert [(a.action, a.price) for a in targets(convert(GOOD_DEAL))] == [("chat", None), ("offer", 92), ("accept", None)]


def test_wait_hours_is_always_zero_because_the_data_has_no_timing():
    assert {a.wait_hours for a in targets(convert(GOOD_DEAL))} == {0}


def test_formal_events_get_default_text():
    assert targets(convert(GOOD_DEAL))[-1].message == "Deal, thanks."


def test_formal_offer_without_text_still_says_something():
    d = dialogue([msg(BUYER, "$85?", "init-price", 85), offer(SELLER, 90), formal(BUYER, "accept")], deal=90)
    assert targets(convert(d))[0].message == "I can do $90."


def test_prompts_match_what_the_live_agent_sends():
    """The whole point of replaying: the prompt is the one PromptedSeller builds in a real game."""
    first = convert(GOOD_DEAL)[0]
    assert first[0]["role"] == "system" and first[0]["content"].startswith("You are selling a second-hand item")
    assert "## Honesty" in first[0]["content"]  # honesty_instruction's default, as in the experiments
    assert "[Status: your current offer $100; their current offer none yet]" in first[1]["content"]
    assert "[Buyer's action: chat / message]" in first[1]["content"]
    assert PromptedSeller.prompt_file == "seller_a.md"


def test_only_the_last_user_message_has_status_lines():
    last = convert(GOOD_DEAL)[-1]
    users = [m["content"] for m in last if m["role"] == "user"]
    assert "[Status:" in users[-1] and not any("[Status:" in u for u in users[:-1])
    assert "their current offer $90" in users[-1]


def test_buyer_private_target_never_appears_in_a_prompt():
    d = dialogue(GOOD_DEAL["events"], deal=90, buyer_target=63)
    text = json.dumps(convert(d))
    assert "63" not in text


def test_floor_is_in_the_prompt_and_never_above_the_deal():
    for i in range(30):
        d = dialogue(GOOD_DEAL["events"], deal=90, uuid=f"C_{i}")
        system = convert(d)[0][0]["content"]
        floor = float(system.split("(your floor) is SGD ")[1].split(".\n")[0])
        assert 70 <= floor <= 90


def test_floor_is_reproducible():
    assert convert(GOOD_DEAL) == convert(GOOD_DEAL)


def test_seller_speaking_first_gets_a_buyer_greeting():
    d = dialogue([
        msg(SELLER, "Hi, are you interested?", "intro"),
        msg(BUYER, "$88?", "init-price", 88),
        formal(SELLER, "accept"),
    ], deal=88)
    examples = convert(d)
    assert BUYER_OPENER in examples[0][1]["content"]
    assert targets(examples)[0].message == "Hi, are you interested?"


def test_consecutive_events_by_one_side_merge_into_one_turn():
    d = dialogue([
        msg(BUYER, "Hello.", "intro"),
        msg(BUYER, "$88?", "init-price", 88),
        msg(SELLER, "Ok.", "agree"),
        formal(SELLER, "accept"),
    ], deal=88)
    examples = convert(d)
    assert "Hello. $88?" in examples[0][1]["content"]
    assert len(examples) == 1 and targets(examples)[0].message == "Ok."


def test_verbal_agreement_at_the_closing_price_is_an_accept():
    d = dialogue([
        msg(BUYER, "$88?", "init-price", 88),
        msg(SELLER, "Sure, that works.", "agree"),
        offer(BUYER, 88),
        formal(SELLER, "accept"),
    ], deal=88)
    assert [a.action for a in targets(convert(d))] == ["accept"]


def test_formal_offer_beats_what_the_text_said():
    d = dialogue([
        msg(BUYER, "$85?", "init-price", 85),
        msg(SELLER, "How about 95?", "counter-price", 95),
        offer(SELLER, 90),
        formal(BUYER, "accept"),
    ], deal=90)
    assert targets(convert(d))[0].price == 90


def test_refusing_an_offer_below_the_floor_is_kept_as_a_good_outcome():
    d = dialogue([
        msg(BUYER, "$45?", "init-price", 45),
        msg(SELLER, "Far too low, I could do $95.", "counter-price", 95),
        msg(BUYER, "$50, final.", "insist", 50),
        formal(SELLER, "quit"),
    ])
    examples = convert(d)
    assert targets(examples)[-1].action == "walk_away"
    floor = float(examples[0][0]["content"].split("(your floor) is SGD ")[1].split(".\n")[0])
    assert floor > 50


@pytest.mark.parametrize("name, d, reason", [
    ("housing", dialogue(GOOD_DEAL["events"], deal=90, category="housing"), "category"),
    ("poor deal", dialogue(GOOD_DEAL["events"], deal=70), "reward_filter"),
    ("buyer quit", dialogue([msg(BUYER, "$45?", "init-price", 45), msg(SELLER, "No.", "disagree"), formal(BUYER, "quit")]), "reward_filter"),
    ("refusal with no offer", dialogue([msg(BUYER, "Hi", "intro"), formal(SELLER, "quit")]), "reward_filter"),
    ("mis-parsed price", dialogue([msg(BUYER, "$88?", "init-price", 88), msg(SELLER, "It's a 2015 model.", "counter-price", 2015), formal(BUYER, "quit")]), "reward_filter"),
    ("price out of range", dialogue([msg(BUYER, "$88?", "init-price", 88), msg(SELLER, "$500", "counter-price", 500), msg(BUYER, "ok", "agree", 90), formal(SELLER, "accept")], deal=90), "price_out_of_range"),
    ("seller raises", dialogue([msg(BUYER, "$80?", "init-price", 80), msg(SELLER, "$92", "counter-price", 92), msg(BUYER, "$85", "counter-price", 85), msg(SELLER, "$96", "counter-price", 96), offer(BUYER, 90), formal(SELLER, "accept")], deal=90), "seller_raised_price"),
    ("seller below deal", dialogue([msg(BUYER, "$80?", "init-price", 80), msg(SELLER, "$85", "counter-price", 85), offer(BUYER, 90), formal(SELLER, "accept")], deal=90), "seller_offered_below_deal"),
    ("price only in text", dialogue([msg(BUYER, "$88?", "init-price", 88), msg(SELLER, "I'd part with it for 95", "unknown"), offer(BUYER, 90), formal(SELLER, "accept")], deal=90), "price_said_in_chat"),
    ("silent seller", dialogue([msg(BUYER, "$88?", "init-price", 88), msg(SELLER, "", "unknown"), offer(BUYER, 90), formal(SELLER, "accept")], deal=90), "silent_seller_turn"),
    ("no annotations", dialogue([{"agent": BUYER, "action": "message", "data": "$88?", "metadata": None}, formal(SELLER, "accept")], deal=88), "no_annotations"),
])
def test_unusable_dialogues_are_skipped_with_a_reason(name, d, reason):
    with pytest.raises(Skip, match=reason):
        convert(d)


def test_a_seller_stating_its_floor_is_dropped():
    # Find a floor for this dialogue, then have the seller say it out loud in chat.
    system = convert(GOOD_DEAL)[0][0]["content"]
    floor = system.split("(your floor) is SGD ")[1].split(".\n")[0]
    d = dialogue([msg(BUYER, "$85?", "init-price", 85), msg(SELLER, f"My lowest is {floor}.", "inform", None), offer(BUYER, 90), formal(SELLER, "accept")], deal=90)
    with pytest.raises(Skip, match="price_said_in_chat|states_floor"):
        convert(d)


def test_contact_details_are_removed():
    assert scrub("email me at sam281@gmail.com or call 415-555-0199") == "email me at [removed] or call [removed]"
    assert scrub("It has 34,000 miles, $90") == "It has 34,000 miles, $90"


def test_build_split_reports_the_funnel(tmp_path):
    out = tmp_path / "x.jsonl"
    report = build_split([GOOD_DEAL, dialogue(GOOD_DEAL["events"], deal=90, category="housing")], out)
    assert report["dialogues"] == 2
    assert report["funnel"] == {"kept": 1, "kept_deal": 1, "category": 1}
    assert report["moves"] == {"chat": 1, "offer": 1, "accept": 1} and report["examples"] == 3
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["C_1-0", "C_1-1", "C_1-2"]
    assert rows[-1]["messages"][-1]["role"] == "assistant"
