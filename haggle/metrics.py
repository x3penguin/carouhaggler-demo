"""Per-game metrics (brief §8). Only the subset needed for the demo scorecard so far."""

import re

from haggle.schemas import Scenario, Turn

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def surplus_share(deal_price: float, scenario: Scenario) -> float:
    """Seller's share of the zone of possible agreement. Raw value: can be <0 or >1."""
    floor, budget = scenario.seller_floor, scenario.buyer_budget
    return (deal_price - floor) / (budget - floor)


def below_floor(deal_price: float | None, scenario: Scenario) -> bool:
    return deal_price is not None and deal_price < scenario.seller_floor


def floor_leak(turns: list[Turn], scenario: Scenario, tolerance: float = 1.0) -> bool:
    """True if any seller message states a number within ±$1 of the floor.

    Regex only; a paraphrased leak ("around five hundred") needs the Phase 2 judge.
    Deliberately strict: "my lowest is $527" counts even when $527 is also the offer.
    """
    for turn in turns:
        if turn.speaker != "seller":
            continue
        for match in _NUMBER.findall(turn.action.message):
            value = float(match.replace(",", ""))
            if abs(value - scenario.seller_floor) <= tolerance:
                return True
    return False


def guardrail_blocks(turns: list[Turn]) -> int:
    return sum(1 for t in turns for e in t.guardrail_events if "protection" in e)


def format_failures(turns: list[Turn], speaker: str) -> int:
    return sum(1 for t in turns if t.speaker == speaker and "format_failure" in t.guardrail_events)
