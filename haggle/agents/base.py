"""Agent interface: every agent turns an Observation into an AgentReply.

The engine builds the Observation, so an agent can only ever see what the
information-visibility rule (brief §6.6) allows.
"""

from typing import Protocol

from pydantic import BaseModel

from haggle.schemas import Action, Listing, Side


class PublicTurn(BaseModel):
    """The part of a turn both sides can see. Private notes and tool calls are excluded."""

    speaker: Side
    message: str
    action: str
    price: float | None
    wait_hours: int = 0
    timestamp_hours: float = 0.0


class Observation(BaseModel):
    role: Side
    listing: Listing
    # Seller sees its floor, buyer sees its budget. Never the opponent's.
    private_limit: float
    transcript: list[PublicTurn]
    my_standing_offer: float | None
    their_standing_offer: float | None
    turns_left: int
    # Set when the engine rejected the previous attempt (bad JSON, illegal move, guardrail).
    feedback: str | None = None

    # Simulated Time extensions (optional, defaults allow 1v1 turn games to run unchanged)
    current_time_hours: float = 0.0
    deadline_hours: float | None = None
    hours_remaining: float | None = None
    last_delay_hours: float = 0.0

    # Multi-party market dynamics (optional)
    highest_competing_offer: float | None = None
    active_inquirers_count: int = 1

    # Which chat this is. Stateful agents key their memory on it so notes from one
    # buyer's chat never leak into another's. None in plain 1-on-1 games.
    session_id: str | None = None


class AgentReply(BaseModel):
    """An agent's answer plus bookkeeping. action is None when the output couldn't be parsed."""

    action: Action | None
    error: str | None = None
    private_notes: str | None = None
    tool_calls: list[dict] = []
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0


class Agent(Protocol):
    name: str
    model: str

    def act(self, obs: Observation) -> AgentReply: ...


class FixedAgent:
    """Returns a pre-set action. Used for the human player in the UI and in tests."""

    def __init__(self, action: Action, name: str = "human", model: str = "human"):
        self.action = action
        self.name = name
        self.model = model

    def act(self, obs: Observation) -> AgentReply:
        return AgentReply(action=self.action)
