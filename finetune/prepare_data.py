"""CraigslistBargain -> chat-format training data for the fine-tuned seller (RQ2, Agents C/D).

Run from the repo root:  uv run python -m finetune.prepare_data

Three decisions shape the output, and each one is about training the model on what it will
actually see in a game:

1. Prompts come from the real code. Every dialogue is replayed through the engine with
   PromptedSeller building the messages, so a training prompt is identical to a live one and
   every kept dialogue is a legal game under our rules.

2. Reward filter. We keep dialogues where the seller did well, so the model imitates good
   negotiators rather than average ones: deals in the top part of the buyer-target-to-asking
   range, plus no-deals where the seller refused an offer below its floor. Without the second
   group the model would never see a refusal and could only learn to close.

3. Missing fields are filled in or left neutral, never invented as behaviour:
   - No seller floors exist in the data, so we draw one the way scenarios.make_scenario does and
     cap it so the seller's real moves respect it.
   - No timing exists, so wait_hours is always 0 and the model does not learn pacing from data.
   - No condition facts exist, so the prompt points at the description.
   - Sellers sometimes speak first, but our engine has the buyer open, so those get a neutral
     buyer greeting.

Only the train and dev splits are used. The published test split has no price or intent
annotations, so its offers can't be told apart from chat. Dev is our held-out set, written as valid.jsonl
because MLX requires that name.
"""

import argparse
import json
import random
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from haggle.agents.base import AgentReply
from haggle.agents.llm_agent import money
from haggle.agents.prompted_seller import PromptedSeller
from haggle.engine import Game
from haggle.guardrails import states_limit
from haggle.schemas import Action, Listing, Scenario

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = "https://worksheets.codalab.org/rest/bundles"
SPLIT_URLS = {
    "train": f"{BUNDLES}/0xd34bbbc5fb3b4fccbd19e10756ca8dd7/contents/blob/parsed.json",
    "dev": f"{BUNDLES}/0x15c4160b43d44ee3a8386cca98da138c/contents/blob/parsed.json",
}

# Renting a flat is not a second-hand sale, and none of our listings are housing.
DROP_CATEGORIES = {"housing"}
# Deal price as a share of the way from the buyer's target to the asking price. 0.5 keeps roughly
# the best 30% of deals. The buyer's target is used only here, never shown in a prompt.
MIN_SELLER_SCORE = 0.5
PRICE_INTENTS = {"init-price", "counter-price", "insist", "vague-price"}
# Offers outside these shares of the asking price are nearly always mis-parsed numbers.
MIN_PRICE_SHARE, MAX_PRICE_SHARE = 0.4, 1.05
MAX_DESCRIPTION_CHARS = 800

BUYER_OPENER = "Hi, is this still available?"
NEUTRAL_FACTS = ["No condition details beyond the description."]
# The dataset's formal accept/reject/quit events carry no text.
DEFAULT_TEXT = {"accept": "Deal, thanks.", "walk_away": "Sorry, I'll pass.", "reject": "Sorry, no."}
# Craigslist text sometimes has contact details left in it. They have no place in a training set.
_CONTACT = re.compile(r"\S+@\S+|\+?\d[\d\s().-]{7,}\d")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass
class RawTurn:
    """One side's consecutive dataset events merged into a single turn (our engine alternates)."""

    side: str
    text: str = ""
    formal: str | None = None  # offer, accept, reject or quit
    formal_price: float | None = None
    said_price: float | None = None
    agreed: bool = False


class Skip(Exception):
    """A dialogue we can't use. The message is its line in the funnel report."""


def scrub(text: str) -> str:
    return _CONTACT.sub("[removed]", text)


def merge_turns(dialogue: dict) -> list[RawTurn]:
    roles = [kb["personal"]["Role"] for kb in dialogue["scenario"]["kbs"]]
    turns: list[RawTurn] = []
    for event in dialogue["events"]:
        side = roles[event["agent"]]
        if not turns or turns[-1].side != side:
            turns.append(RawTurn(side))
        turn, meta = turns[-1], event.get("metadata") or {}
        if event["action"] == "message":
            if not event.get("metadata"):
                raise Skip("no_annotations")  # can't tell whether this message makes an offer
            turn.text = f"{turn.text} {scrub(str(event['data'] or '').strip())}".strip()
            if meta.get("intent") in PRICE_INTENTS and meta.get("price"):
                turn.said_price = float(meta["price"])
            turn.agreed = turn.agreed or meta.get("intent") == "agree"
        elif event["action"] == "offer":
            turn.formal, turn.formal_price = "offer", float((event["data"] or {}).get("price") or 0)
        elif event["action"] in ("accept", "reject", "quit"):
            turn.formal = event["action"]
    return turns


def to_action(turn: RawTurn, buyer_standing: float | None, deal: float | None) -> Action:
    """The move this turn makes in our engine's terms. Formal events beat what the text implied."""
    move, price = "chat", None
    if turn.formal == "offer":
        move, price = "offer", turn.formal_price
    elif turn.formal == "accept":
        move = "accept"
    elif turn.formal == "quit":
        move = "walk_away"
    elif turn.formal == "reject":
        move = "reject"
    elif turn.side == "seller" and turn.agreed and deal is not None and buyer_standing == deal:
        move = "accept"  # the seller said yes to the price the deal closed at
    elif turn.said_price is not None:
        move, price = "offer", turn.said_price
    default = f"I can do ${money(price)}." if move == "offer" else DEFAULT_TEXT.get(move, "")
    return Action(message=turn.text.rstrip("\\ ") or default, action=move, price=price)


def plan_moves(dialogue: dict) -> tuple[list[Action], float | None]:
    """Moves in turn order, buyer first, stopping at the move that ends the game; plus the deal price."""
    outcome = dialogue["outcome"] or {}
    deal = float(outcome["offer"]["price"]) if outcome.get("reward") == 1 and outcome.get("offer") else None
    turns = merge_turns(dialogue)
    if not turns:
        raise Skip("empty")
    if turns[0].side == "seller":
        turns.insert(0, RawTurn("buyer", text=BUYER_OPENER))
    moves: list[Action] = []
    buyer_standing = None
    for turn in turns:
        action = to_action(turn, buyer_standing, deal)
        moves.append(action)
        if turn.side == "buyer" and action.action == "offer":
            buyer_standing = action.price
        if action.action in ("accept", "walk_away"):
            break
    return moves, deal


def names_a_price(message: str, asking: float) -> bool:
    """True if the text has a number in the plausible price range for this item."""
    numbers = (float(n.replace(",", "")) for n in _NUMBER.findall(message))
    return any(MIN_PRICE_SHARE * asking <= n <= MAX_PRICE_SHARE * asking for n in numbers)


def check_moves(moves: list[Action], asking: float, deal: float | None) -> None:
    """Reject dialogues that would teach the wrong habit or whose annotations look wrong."""
    for action in moves:
        if action.action == "offer" and not MIN_PRICE_SHARE * asking <= action.price <= MAX_PRICE_SHARE * asking:
            raise Skip("price_out_of_range")
    for action in moves[1::2]:
        if action.action == "chat" and not action.message:
            raise Skip("silent_seller_turn")  # a data gap, not a tactic
        if action.action == "chat" and names_a_price(action.message, asking):
            raise Skip("price_said_in_chat")  # the seller named a price but no offer was recorded
    seller_offers = [m.price for m in moves[1::2] if m.action == "offer"]
    if any(later > earlier for earlier, later in zip(seller_offers, seller_offers[1:])):
        raise Skip("seller_raised_price")
    if deal is not None and any(p < deal for p in seller_offers):
        raise Skip("seller_offered_below_deal")


def is_good_outcome(moves: list[Action], deal: float | None, asking: float, buyer_target: float) -> bool:
    if deal is not None:
        return (deal - buyer_target) / (asking - buyer_target) >= MIN_SELLER_SCORE
    # A no-deal counts only if the seller was the one who refused an actual offer.
    seller_moves, buyer_moves = moves[1::2], moves[0::2]
    return (
        bool(seller_moves)
        and len(moves) % 2 == 0  # the seller moved last
        and seller_moves[-1].action in ("walk_away", "reject")
        and any(m.action == "offer" for m in buyer_moves)
    )


def draw_floor(dialogue_id: str, moves: list[Action], asking: float, deal: float | None) -> float:
    """A private floor drawn like make_scenario's (70-90% of asking), consistent with the moves.

    The seller never went below it. For a no-deal, the buyer's best offer stays below it, so
    the refusal in the data is the correct call.
    """
    rng = random.Random(f"{dialogue_id}:floor")
    floor = float(round(asking * rng.uniform(0.70, 0.90)))
    seller_offers = [m.price for m in moves[1::2] if m.action == "offer"]
    highest = min(seller_offers + [asking] + ([deal] if deal is not None else []))
    if deal is not None:
        return min(floor, highest)
    best_buyer = max(m.price for m in moves[0::2] if m.action == "offer")
    floor = min(max(floor, best_buyer + 1), highest)
    if floor <= best_buyer:
        raise Skip("no_consistent_floor")
    return floor


class _NoClient:
    """PromptedSeller wants a client for its model name. Nothing is ever called here."""

    model = "dataset"


class _Replay(PromptedSeller):
    """Plays the seller's recorded moves and keeps the exact prompt behind each one."""

    def __init__(self, moves: list[Action]):
        super().__init__(_NoClient())
        self.moves = iter(moves)
        self.examples: list[list[dict]] = []

    def act(self, obs) -> AgentReply:
        action = next(self.moves)
        target = {"role": "assistant", "content": action.model_dump_json()}
        self.examples.append(self.messages(obs) + [target])
        return AgentReply(action=action)


class _ScriptedBuyer:
    name, model = "dataset_buyer", "dataset"

    def __init__(self, moves: list[Action]):
        self.moves = iter(moves)

    def act(self, obs) -> AgentReply:
        return AgentReply(action=next(self.moves))


def convert(dialogue: dict) -> list[list[dict]]:
    """Replay one dialogue through the engine. Returns the seller's training examples or raises Skip."""
    kbs = dialogue["scenario"]["kbs"]
    seller_kb = next(kb for kb in kbs if kb["personal"]["Role"] == "seller")
    buyer_kb = next(kb for kb in kbs if kb["personal"]["Role"] == "buyer")
    item = seller_kb["item"]
    if dialogue["scenario"]["category"] in DROP_CATEGORIES:
        raise Skip("category")
    asking, buyer_target = float(item["Price"] or 0), float(buyer_kb["personal"]["Target"] or 0)
    if asking <= 0 or buyer_target >= asking:
        raise Skip("no_price_range")

    moves, deal = plan_moves(dialogue)
    if not is_good_outcome(moves, deal, asking, buyer_target):
        raise Skip("reward_filter")
    check_moves(moves, asking, deal)
    floor = draw_floor(dialogue["uuid"], moves, asking, deal)
    if any(states_limit(m, floor) for m in moves[1::2]):
        raise Skip("states_floor")  # don't teach the model to reveal its floor

    listing = Listing(
        id=dialogue["uuid"],
        title=item["Title"],
        category=item["Category"],
        description=scrub(" ".join(item["Description"]))[:MAX_DESCRIPTION_CHARS],
        condition_facts=NEUTRAL_FACTS,
        asking_price=asking,
    )
    buyer_offers = [m.price for m in moves[0::2] if m.action == "offer"]
    scenario = Scenario(
        id=dialogue["uuid"], listing=listing, seller_floor=floor, buyer_persona="dataset", seed=0,
        buyer_budget=max([floor + 1] + buyer_offers),
    )
    game = Game(scenario, max_parse_retries=0, floor_protection=True, budget_protection=False)
    seller, buyer = _Replay(moves[1::2]), _ScriptedBuyer(moves[0::2])
    for _ in moves:
        turn = game.step(buyer if game.next_speaker == "buyer" else seller)
        if turn.guardrail_events:
            raise Skip("illegal_in_engine")
    if deal is not None and (game.outcome != "deal" or game.deal_price != deal):
        raise Skip("deal_mismatch")
    return seller.examples


def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, url in SPLIT_URLS.items():
        path = data_dir / f"{split}.json"
        if not path.exists():
            print(f"downloading {split}")
            urllib.request.urlretrieve(url, path)


def build_split(dialogues: list[dict], out_path: Path) -> dict:
    """Write one JSONL split and return its funnel: why dialogues dropped, and the move mix kept."""
    funnel, moves = Counter(), Counter()
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for dialogue in dialogues:
            try:
                examples = convert(dialogue)
            except Skip as reason:
                funnel[str(reason)] += 1
                continue
            funnel["kept"] += 1
            outcome = "deal" if (dialogue["outcome"] or {}).get("reward") == 1 else "refusal"
            funnel[f"kept_{outcome}"] += 1
            for i, messages in enumerate(examples):
                moves[json.loads(messages[-1]["content"])["action"]] += 1
                f.write(json.dumps({"id": f"{dialogue['uuid']}-{i}", "messages": messages}, ensure_ascii=False) + "\n")
    return {"dialogues": len(dialogues), "funnel": dict(funnel), "examples": sum(moves.values()), "moves": dict(moves)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "craigslist")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "craigslist" / "sft")
    args = parser.parse_args()

    download(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for split in SPLIT_URLS:
        dialogues = json.loads((args.data_dir / f"{split}.json").read_text(encoding="utf-8"))
        name = "valid" if split == "dev" else split
        report[name] = build_split(dialogues, args.out_dir / f"{name}.jsonl")
    (args.out_dir / "funnel.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
