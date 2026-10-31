"""MultiGameOrchestrator: coordinates concurrent buyer negotiations over a single listing.

Simulates virtual time (hours/days) using an event queue without any real-world sleep.
When an item is sold in any session, all other sessions terminate with `no_deal_item_sold`.
"""

from dataclasses import dataclass, field
import heapq
from typing import Literal

from haggle.agents.base import Agent
from haggle.engine import Decision, Game, other
from haggle.schemas import GameResult, Listing, Scenario, Side


@dataclass(order=True)
class _Event:
    timestamp_hours: float
    event_id: int
    session_id: str = field(compare=False)
    speaker: Side = field(compare=False)
    # Set for a delivery event: the decision made earlier, arriving after its wait_hours.
    decision: Decision | None = field(default=None, compare=False)


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
    """One seller, several buyers, one shared clock.

    Each event is either a decision (an agent chooses its next message) or, when that
    message has wait_hours > 0, its later delivery. Nothing a message does (a new offer,
    a deal) is visible to anyone until it is delivered.
    """

    def __init__(
        self,
        listing: Listing,
        deadline_days: float = 7.0,
        enforce_floor_protection: bool = True,
        enforce_price_direction: bool = False,
        max_turns_per_side: int = 10,
        buyers_see_competing_offers: bool = True,
    ):
        self.listing = listing
        self.deadline_days = float(deadline_days)
        self.deadline_hours = float(deadline_days * 24.0)
        self.enforce_floor_protection = enforce_floor_protection
        self.enforce_price_direction = enforce_price_direction
        self.max_turns_per_side = max_turns_per_side
        # Team decision (ROLE_SPLIT item 4): real marketplaces hide rival offers from buyers.
        self.buyers_see_competing_offers = buyers_see_competing_offers

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

    def _live(self, session: Session) -> bool:
        return session.arrival_hours <= self.simulated_time_hours and not session.game.is_over

    def highest_competing_offer(self, exclude_session_id: str | None = None) -> float | None:
        """Best delivered buyer offer among other chats that are still open."""
        offers = [
            s.game.standing["buyer"]
            for s_id, s in self.sessions.items()
            if s_id != exclude_session_id and self._live(s) and s.game.standing["buyer"] is not None
        ]
        return max(offers, default=None)

    def active_sessions_count(self) -> int:
        """Buyers who have arrived and whose chat is still open."""
        return sum(1 for s in self.sessions.values() if self._live(s))

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
        for other_session in self.sessions.values():
            if other_session.scenario.seller_floor != scenario.seller_floor:
                raise ValueError(
                    "All sessions must share the seller's floor: it is one seller with one item "
                    f"(got {scenario.seller_floor}, expected {other_session.scenario.seller_floor})."
                )

        game = Game(
            scenario=scenario,
            max_turns_per_side=self.max_turns_per_side,
            floor_protection=self.enforce_floor_protection,
            enforce_price_direction=self.enforce_price_direction,
            seller_deadline_days=self.deadline_days,
            buyer_deadline_days=scenario.buyer_deadline_days,
            start_hours=arrival_hours,
        )
        self.sessions[session_id] = Session(
            session_id=session_id,
            scenario=scenario,
            buyer=buyer,
            seller=seller,
            game=game,
            arrival_hours=arrival_hours,
        )
        self._schedule(arrival_hours, session_id, "buyer")

    def _schedule(self, timestamp_hours: float, session_id: str, speaker: Side, decision: Decision | None = None) -> None:
        self._event_counter += 1
        heapq.heappush(
            self.event_queue,
            _Event(float(timestamp_hours), self._event_counter, session_id, speaker, decision),
        )

    def _close_others(self, winner: str) -> None:
        for s_id, s in self.sessions.items():
            if s_id != winner and not s.game.is_over:
                s.game.terminate_item_sold(f"Item sold to {winner} at S${self.deal_price:.2f}")

    def step(self) -> bool:
        """Processes the next scheduled event. Returns True if more events remain, False if done."""
        if not self.event_queue:
            return False

        event = heapq.heappop(self.event_queue)
        self.simulated_time_hours = event.timestamp_hours
        session = self.sessions[event.session_id]
        game = session.game

        if self.simulated_time_hours >= self.deadline_hours:
            self.inventory_status = "expired"
            for s in self.sessions.values():
                if not s.game.is_over:
                    s.game.terminate_deadline(f"Listing expired at day {self.simulated_day:.1f}")
            self.event_queue.clear()
            return False

        if self.inventory_status == "sold":
            if not game.is_over:
                game.terminate_item_sold(f"Item sold to {self.sold_to} at S${self.deal_price:.2f}")
            return bool(self.event_queue)

        if game.is_over:
            return bool(self.event_queue)

        if event.decision is not None:
            turn = game.commit(event.decision)
        else:
            side = event.speaker
            agent = session.buyer if side == "buyer" else session.seller
            game.current_time_hours = self.simulated_time_hours
            competing = self.highest_competing_offer(exclude_session_id=event.session_id)
            if side == "buyer" and not self.buyers_see_competing_offers:
                competing = None
            base = game.observation(
                side=side,
                current_time_hours=self.simulated_time_hours,
                highest_competing_offer=competing,
                active_inquirers_count=self.active_sessions_count(),
            )
            # Merged into every observation the engine builds, including retries and re-asks.
            context = {
                "current_time_hours": base.current_time_hours,
                "deadline_hours": base.deadline_hours,
                "hours_remaining": base.hours_remaining,
                "highest_competing_offer": competing,
                "active_inquirers_count": base.active_inquirers_count,
                "session_id": event.session_id,
            }
            decision = game.decide(_WithMarketContext(agent, context))
            if decision.action.wait_hours > 0:
                self._schedule(self.simulated_time_hours + decision.action.wait_hours, event.session_id, side, decision)
                return True
            turn = game.commit(decision)

        if game.outcome == "deal":
            self.inventory_status = "sold"
            self.sold_to = event.session_id
            self.deal_price = game.deal_price
            self._close_others(event.session_id)
        elif not game.is_over:
            self._schedule(turn.simulated_time_hours, event.session_id, other(turn.speaker))
        return bool(self.event_queue)

    def run(self, config_hash: str = "") -> dict[str, GameResult]:
        """Runs the multi-session simulation to completion."""
        while self.step():
            pass
        return {s_id: s.game.result(s.buyer, s.seller, config_hash) for s_id, s in self.sessions.items()}
