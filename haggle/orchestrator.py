"""MultiGameOrchestrator: coordinates concurrent buyer negotiations over a single listing.

Simulates virtual time (hours/days) using an event queue without any real-world sleep.
When an item is sold in any session, all other sessions terminate with `no_deal_item_sold`.
"""

from dataclasses import dataclass, field
import heapq
from typing import Literal

from haggle.agents.base import Agent
from haggle.engine import Game, other
from haggle.schemas import GameResult, Listing, Scenario, Side


@dataclass(order=True)
class _Event:
    timestamp_hours: float
    event_id: int
    session_id: str = field(compare=False)
    speaker: Side = field(compare=False)


class _WithMarketContext:
    """Forwards each engine call to the real agent with the orchestrator's context added.

    The engine may call act() several times in one step (parse retries, guardrail re-asks),
    each time with fresh feedback. Calling the agent again on each of those, rather than
    replaying its first reply, is what lets retries and guardrails work in multi-buyer mode,
    and keeps token counts to one per real model call.
    """

    def __init__(self, agent: Agent, context: dict):
        self.agent = agent
        self.name = agent.name
        self.model = agent.model
        self.context = context

    def act(self, obs):
        return self.agent.act(obs.model_copy(update=self.context))


@dataclass
class Session:
    session_id: str
    scenario: Scenario
    buyer: Agent
    seller: Agent
    game: Game
    arrival_hours: float
    completed: bool = False


class MultiGameOrchestrator:
    def __init__(
        self,
        listing: Listing,
        deadline_days: float = 7.0,
        enforce_floor_protection: bool = True,
        enforce_price_direction: bool = False,
        max_turns_per_side: int = 10,
    ):
        self.listing = listing
        self.deadline_hours = float(deadline_days * 24.0)
        self.enforce_floor_protection = enforce_floor_protection
        self.enforce_price_direction = enforce_price_direction
        self.max_turns_per_side = max_turns_per_side

        self.sessions: dict[str, Session] = {}
        self.event_queue: list[_Event] = []
        self._event_counter = 0

        self.simulated_time_hours: float = 0.0
        self.inventory_status: Literal["available", "sold", "expired"] = "available"
        self.sold_to: str | None = None
        self.deal_price: float | None = None

    @property
    def simulated_day(self) -> float:
        return self.simulated_time_hours / 24.0

    @property
    def hours_remaining(self) -> float:
        return max(0.0, self.deadline_hours - self.simulated_time_hours)

    def highest_competing_offer(self, exclude_session_id: str | None = None) -> float | None:
        """Finds highest buyer standing offer across other active sessions."""
        best: float | None = None
        for s_id, session in self.sessions.items():
            if exclude_session_id is not None and s_id == exclude_session_id:
                continue
            b_offer = session.game.standing.get("buyer")
            if b_offer is not None:
                if best is None or b_offer > best:
                    best = b_offer
        return best

    def active_sessions_count(self) -> int:
        return sum(1 for s in self.sessions.values() if not s.game.is_over)

    def add_session(
        self,
        session_id: str,
        scenario: Scenario,
        buyer: Agent,
        seller: Agent,
        arrival_hours: float = 0.0,
    ) -> None:
        """Registers a buyer entering the marketplace negotiation at `arrival_hours`."""
        if session_id in self.sessions:
            raise ValueError(f"Session '{session_id}' already registered.")
        if scenario.listing.id != self.listing.id:
            raise ValueError("All sessions must share the same listing.")

        game = Game(
            scenario=scenario,
            max_turns_per_side=self.max_turns_per_side,
            floor_protection=self.enforce_floor_protection,
            enforce_price_direction=self.enforce_price_direction,
        )
        session = Session(
            session_id=session_id,
            scenario=scenario,
            buyer=buyer,
            seller=seller,
            game=game,
            arrival_hours=arrival_hours,
        )
        self.sessions[session_id] = session

        # Schedule buyer's initial message
        self._schedule(arrival_hours, session_id, "buyer")

    def _schedule(self, timestamp_hours: float, session_id: str, speaker: Side) -> None:
        self._event_counter += 1
        heapq.heappush(
            self.event_queue,
            _Event(
                timestamp_hours=float(timestamp_hours),
                event_id=self._event_counter,
                session_id=session_id,
                speaker=speaker,
            ),
        )

    def step(self) -> bool:
        """Processes the next scheduled event. Returns True if more events remain, False if done."""
        if not self.event_queue:
            return False

        event = heapq.heappop(self.event_queue)
        self.simulated_time_hours = event.timestamp_hours
        session = self.sessions[event.session_id]
        game = session.game

        # Check deadline
        if self.simulated_time_hours > self.deadline_hours:
            self.inventory_status = "expired"
            for s in self.sessions.values():
                if not s.game.is_over:
                    s.game.terminate_deadline(f"Expired at day {self.simulated_day:.1f}")
            self.event_queue.clear()
            return False

        # If item already sold to another buyer, terminate this session
        if self.inventory_status == "sold":
            if not game.is_over:
                game.terminate_item_sold(f"Item sold to {self.sold_to} at S${self.deal_price:.2f}")
            return bool(self.event_queue)

        if game.is_over:
            return bool(self.event_queue)

        # Contextual observation for speaker
        side = event.speaker
        agent = session.buyer if side == "buyer" else session.seller
        competing = self.highest_competing_offer(exclude_session_id=event.session_id)
        active_count = self.active_sessions_count()

        # The market context the game itself doesn't know about. Merged into every
        # observation the engine builds, including retries and guardrail re-asks.
        base = game.observation(
            side=side,
            current_time_hours=self.simulated_time_hours,
            deadline_hours=self.deadline_hours,
            highest_competing_offer=competing,
            active_inquirers_count=active_count,
        )
        context = {
            "current_time_hours": base.current_time_hours,
            "deadline_hours": base.deadline_hours,
            "hours_remaining": base.hours_remaining,
            "highest_competing_offer": competing,
            "active_inquirers_count": active_count,
            "session_id": event.session_id,
        }
        turn = game.step(_WithMarketContext(agent, context))

        # If deal closed, claim inventory and terminate other sessions
        if game.outcome == "deal":
            self.inventory_status = "sold"
            self.sold_to = event.session_id
            self.deal_price = game.deal_price
            for s_id, s in self.sessions.items():
                if s_id != event.session_id and not s.game.is_over:
                    s.game.terminate_item_sold(
                        f"Item sold to {self.sold_to} at S${self.deal_price:.2f}"
                    )
            return bool(self.event_queue)

        # If not over, schedule the other side's reply
        if not game.is_over:
            delay = float(max(0, turn.action.wait_hours))
            next_time = self.simulated_time_hours + delay
            self._schedule(next_time, event.session_id, other(side))

        return bool(self.event_queue)

    def run(self) -> dict[str, GameResult]:
        """Runs the multi-session simulation to completion."""
        while self.step():
            pass
        return {
            s_id: s.game.result(s.buyer, s.seller)
            for s_id, s in self.sessions.items()
        }
