"""Data schemas shared by the engine, agents, metrics and UI (brief §5).

Everything that crosses a module boundary is a Pydantic model so that LLM output
is validated in one place and game logs round-trip cleanly through JSONL.
"""

from typing import Literal

from pydantic import BaseModel, Field

Side = Literal["buyer", "seller"]
ActionType = Literal["offer", "accept", "reject", "walk_away"]
Outcome = Literal[
    "deal",
    "no_deal_walkaway",
    "no_deal_turn_limit",
    "no_deal_item_sold",
    "no_deal_deadline",
    "error",
]


class Listing(BaseModel):
    id: str
    title: str
    category: str
    description: str
    # Ground truth for honesty checks. Every listing has at least one flaw (brief §9).
    condition_facts: list[str]
    asking_price: float  # SGD


class Scenario(BaseModel):
    id: str
    listing: Listing
    seller_floor: float  # private to seller
    buyer_budget: float  # private to buyer; always > seller_floor so a deal is possible
    buyer_persona: str
    seed: int
    seller_deadline_days: float = 7.0  # private to seller: days to clear item
    buyer_deadline_days: float = 5.0   # private to buyer: days needed to buy item

    @property
    def deadline_days(self) -> float:
        return min(self.seller_deadline_days, self.buyer_deadline_days)


class Action(BaseModel):
    """What every agent returns each turn. Also used as the LLM's JSON output schema."""

    message: str = Field(description="Chat message shown to the other side.")
    action: ActionType = Field(
        description="offer = propose a price; accept = take the other side's current offer; "
        "reject = decline without a new price; walk_away = end the negotiation."
    )
    price: float | None = Field(default=None, description="Required for 'offer', otherwise null.")
    wait_hours: int = Field(
        default=0,
        ge=0,
        description="Simulated hours to wait before sending this message/offer (e.g. 0 to 48).",
    )


class Turn(BaseModel):
    index: int
    speaker: Side
    action: Action
    private_notes: str | None = None  # never shown to the opponent
    tool_calls: list[dict] = []
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    parse_retries: int = 0
    guardrail_events: list[str] = []


class GameResult(BaseModel):
    scenario: Scenario
    seller_agent: str
    buyer_agent: str
    seller_model: str
    buyer_model: str
    turns: list[Turn]
    outcome: Outcome
    deal_price: float | None
    config_hash: str
    # Both fields are defaulted so older games.jsonl files still parse.
    error: str | None = None
    schema_version: int = 1
