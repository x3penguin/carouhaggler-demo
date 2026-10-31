"""Game engine: turn order, legal moves, guardrails and outcome resolution (brief §6).

All negotiation rules live here, in code, so a model can never "decide" that a deal
happened. Agents only propose actions; the engine decides what they mean.
"""

import math
from dataclasses import dataclass, field

from haggle.agents.base import Agent, Observation, PublicTurn
from haggle.schemas import Action, GameResult, Scenario, Side, Turn


def other(side: Side) -> Side:
    return "seller" if side == "buyer" else "buyer"


@dataclass
class Decision:
    """An action chosen and validated but not yet delivered, so wait_hours can pass first."""

    side: Side
    action: Action
    decided_at_hours: float
    private_notes: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    parse_retries: int = 0
    guardrail_events: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class Game:
    def __init__(
        self,
        scenario: Scenario,
        max_turns_per_side: int = 50,
        max_parse_retries: int = 2,
        floor_protection: bool = True,
        budget_protection: bool = True,
        enforce_price_direction: bool = False,
        seller_deadline_days: float | None = None,
        buyer_deadline_days: float | None = None,
        deadline_days: float | None = None,
        start_hours: float = 0.0,
    ):
        self.scenario = scenario
        self.max_turns_per_side = max_turns_per_side
        self.max_parse_retries = max_parse_retries
        # Switchable because baseline runs must be able to *show* the failure (§6.8).
        self.floor_protection = floor_protection
        # Stops LLM buyers paying over budget. Off for human players, who may do as they like.
        self.budget_protection = budget_protection
        # Optional sanity rule: no buyer offer above the seller's standing offer, and no
        # seller offer below the buyer's. Off by default so existing behaviour is unchanged.
        self.enforce_price_direction = enforce_price_direction

        if deadline_days is not None:
            s_days = seller_deadline_days if seller_deadline_days is not None else deadline_days
            b_days = buyer_deadline_days if buyer_deadline_days is not None else deadline_days
        else:
            s_days = seller_deadline_days if seller_deadline_days is not None else getattr(scenario, "seller_deadline_days", None)
            b_days = buyer_deadline_days if buyer_deadline_days is not None else getattr(scenario, "buyer_deadline_days", None)

        self.seller_deadline_days = s_days
        self.buyer_deadline_days = b_days
        # Absolute times on the listing's clock. The seller's deadline counts from the listing
        # going up; the buyer's from when this buyer arrives (start_hours).
        self.seller_deadline_hours = s_days * 24.0 if s_days is not None else None
        self.buyer_deadline_hours = start_hours + b_days * 24.0 if b_days is not None else None

        active_hours = [h for h in (self.seller_deadline_hours, self.buyer_deadline_hours) if h is not None]
        self.deadline_hours = min(active_hours) if active_hours else None
        self.deadline_days = self.deadline_hours / 24.0 if self.deadline_hours is not None else None
        self.current_time_hours: float = start_hours

        self.turns: list[Turn] = []
        # The asking price is the seller's opening offer (§6.5).
        self.standing: dict[Side, float | None] = {
            "buyer": None,
            "seller": scenario.listing.asking_price,
        }
        self.outcome: str | None = None
        self.deal_price: float | None = None
        # Only real failures (e.g. API timeout). Normal endings are described by `outcome`.
        self.error: str | None = None
        self.note: str | None = None

    # ---- state -------------------------------------------------------------

    @property
    def is_over(self) -> bool:
        return self.outcome is not None

    @property
    def next_speaker(self) -> Side:
        # Buyer opens the chat, then strict alternation (§6.1).
        return "buyer" if len(self.turns) % 2 == 0 else "seller"

    def turns_left(self, side: Side) -> int:
        taken = sum(1 for t in self.turns if t.speaker == side)
        return self.max_turns_per_side - taken

    def private_limit(self, side: Side) -> float:
        return self.scenario.seller_floor if side == "seller" else self.scenario.buyer_budget

    def observation(
        self,
        side: Side,
        feedback: str | None = None,
        current_time_hours: float | None = None,
        deadline_hours: float | None = None,
        highest_competing_offer: float | None = None,
        active_inquirers_count: int = 1,
    ) -> Observation:
        """Everything `side` is allowed to see: public transcript + its own limit only (§6.6)."""
        curr_time = self.current_time_hours if current_time_hours is None else current_time_hours
        if deadline_hours is not None:
            dead_hours = deadline_hours
        else:
            dead_hours = self.seller_deadline_hours if side == "seller" else self.buyer_deadline_hours
        hours_remaining = max(0.0, dead_hours - curr_time) if dead_hours is not None else None
        last_delay = self.turns[-1].action.wait_hours if self.turns else 0.0
        return Observation(
            role=side,
            listing=self.scenario.listing,
            private_limit=self.private_limit(side),
            transcript=[
                PublicTurn(
                    speaker=t.speaker,
                    message=t.action.message,
                    action=t.action.action,
                    price=t.action.price if t.action.action == "offer" else None,
                    wait_hours=t.action.wait_hours,
                    timestamp_hours=t.simulated_time_hours,
                )
                for t in self.turns
            ],
            my_standing_offer=self.standing[side],
            their_standing_offer=self.standing[other(side)],
            turns_left=self.turns_left(side),
            feedback=feedback,
            current_time_hours=curr_time,
            deadline_hours=dead_hours,
            hours_remaining=hours_remaining,
            last_delay_hours=last_delay,
            highest_competing_offer=highest_competing_offer,
            active_inquirers_count=active_inquirers_count,
        )

    def terminate_item_sold(self, note: str = "") -> None:
        """Ends game when item was sold to another concurrent buyer."""
        if not self.is_over:
            self.outcome = "no_deal_item_sold"
            self.note = note or None

    def terminate_deadline(self, note: str = "") -> None:
        """Ends game when simulated time deadline has elapsed."""
        if not self.is_over:
            self.outcome = "no_deal_deadline"
            self.note = note or None

    # ---- validation --------------------------------------------------------

    def rule_error(self, side: Side, action: Action) -> str | None:
        """Moves that are illegal under any settings. Returns a message for the agent, or None."""
        if action.action == "offer" and (
            action.price is None or not math.isfinite(action.price) or action.price <= 0
        ):
            return "An 'offer' needs a positive, finite price."
        if action.action == "offer" and self.enforce_price_direction:
            their = self.standing[other(side)]
            if their is not None:
                if side == "buyer" and action.price > their:
                    return f"Your offer ${action.price:.2f} is above the seller's current offer ${their:.2f}."
                if side == "seller" and action.price < their:
                    return f"Your offer ${action.price:.2f} is below the buyer's current offer ${their:.2f}."
        if action.action == "accept" and self.standing[other(side)] is None:
            return (
                f"The {other(side)} has not made an offer yet, so there is nothing to accept. "
                "If you are chatting, answering questions, or discussing details, use action 'chat'."
            )
        return None

    def guardrail_violation(self, side: Side, action: Action) -> str | None:
        """Moves that are legal but blocked by a guardrail. Returns a log event, or None."""
        limit = self.private_limit(side)
        if side == "seller" and self.floor_protection:
            if action.action == "offer" and action.price < limit:
                return f"floor_protection: blocked offer ${action.price:.2f} < floor ${limit:.2f}"
            their = self.standing["buyer"]
            if action.action == "accept" and their is not None and their < limit:
                return f"floor_protection: blocked accept of ${their:.2f} < floor ${limit:.2f}"
        if side == "buyer" and self.budget_protection:
            if action.action == "offer" and action.price > limit:
                return f"budget_protection: blocked offer ${action.price:.2f} > budget ${limit:.2f}"
            their = self.standing["seller"]
            if action.action == "accept" and their is not None and their > limit:
                return f"budget_protection: blocked accept of ${their:.2f} > budget ${limit:.2f}"
        return None

    def _guardrail_feedback(self, side: Side, action: Action) -> str:
        """Tell the agent why it was blocked so it can choose a valid alternative."""
        limit = self.private_limit(side)
        # No number here: models tend to repeat feedback to the other side, which would leak it.
        if side == "seller":
            return (
                "BLOCKED. That price is below your floor. Counter at or above your floor, "
                "decline, or walk away. Do not mention your floor to the buyer."
            )
        return (
            "BLOCKED. That price is above your budget. Counter at or below your budget, "
            "decline, or walk away. Do not mention your budget to the seller."
        )

    # ---- the turn loop -----------------------------------------------------

    def step(self, agent: Agent) -> Turn:
        """Ask the next speaker for an action and deliver it immediately (after its wait_hours)."""
        return self.commit(self.decide(agent))

    def decide(self, agent: Agent) -> Decision:
        """Ask the next speaker for an action and enforce the rules, without applying it."""
        if self.is_over:
            raise RuntimeError("Game is already over.")
        side = self.next_speaker
        feedback: str | None = None
        retries = 0
        reasked_after_guardrail = False
        events: list[str] = []
        usage = {"tokens_in": 0, "tokens_out": 0, "latency_s": 0.0, "cost_usd": 0.0}
        notes: str | None = None
        tool_calls: list[dict] = []

        while True:
            reply = agent.act(self.observation(side, feedback))
            for key in usage:
                usage[key] += getattr(reply, key)
            notes, tool_calls = reply.private_notes, reply.tool_calls
            events.extend(reply.guardrail_events)

            error = reply.error if reply.action is None else self.rule_error(side, reply.action)
            if error:
                # Parse failures and illegal moves get a retry with the reason (§6.7).
                if retries < self.max_parse_retries:
                    retries += 1
                    feedback = f"Your last reply was invalid: {error} Please reply again."
                    continue
                events.append("format_failure")
                action = Action(message="", action="reject", price=None)
                break

            action = reply.action
            violation = self.guardrail_violation(side, action)
            if violation:
                events.append(violation)
                # One re-ask, then the engine forces a reject (§6.8).
                if not reasked_after_guardrail:
                    reasked_after_guardrail = True
                    feedback = self._guardrail_feedback(side, action)
                    continue
                events.append("guardrail_forced_reject")
                action = Action(message="", action="reject", price=None)
            break

        return Decision(side, action, self.current_time_hours, notes, tool_calls, retries, events, usage)

    def commit(self, d: Decision) -> Turn:
        """Deliver a decision: advance the clock by its wait_hours, log the turn, apply it."""
        if self.is_over:
            raise RuntimeError("Game is already over.")
        self.current_time_hours = d.decided_at_hours + d.action.wait_hours
        turn = Turn(
            index=len(self.turns),
            speaker=d.side,
            action=d.action,
            private_notes=d.private_notes,
            tool_calls=d.tool_calls,
            parse_retries=d.parse_retries,
            guardrail_events=d.guardrail_events,
            simulated_time_hours=self.current_time_hours,
            **d.usage,
        )
        self.turns.append(turn)
        self._apply(d.side, d.action)
        return turn

    def _apply(self, side: Side, action: Action) -> None:
        # A message that arrives at or after a deadline is too late to count, accept included.
        if self.deadline_hours is not None and self.current_time_hours >= self.deadline_hours:
            self.outcome = "no_deal_deadline"
            expired = "buyer" if self.buyer_deadline_hours == self.deadline_hours else "seller"
            self.note = f"{expired} deadline passed at day {self.current_time_hours / 24.0:.1f}"
            return
        if action.action == "offer":
            self.standing[side] = action.price
        elif action.action == "accept":
            # A deal only happens by accepting the other side's standing offer (§6.4).
            self.deal_price = self.standing[other(side)]
            self.outcome = "deal"
        elif action.action == "walk_away":
            self.outcome = "no_deal_walkaway"

        if self.outcome is None and self.turns_left("buyer") <= 0 and self.turns_left("seller") <= 0:
            self.outcome = "no_deal_turn_limit"

    def totals(self) -> dict[Side, dict[str, float]]:
        """Summed tokens, latency and cost per side, so callers need not re-aggregate turns."""
        out = {s: {"tokens_in": 0, "tokens_out": 0, "latency_s": 0.0, "cost_usd": 0.0} for s in ("buyer", "seller")}
        for t in self.turns:
            for key in out[t.speaker]:
                out[t.speaker][key] += getattr(t, key)
        return out

    def play(self, buyer: Agent, seller: Agent) -> None:
        """Run to completion with two automated agents."""
        while not self.is_over:
            try:
                self.step(buyer if self.next_speaker == "buyer" else seller)
            except Exception as e:  # one failed call must not kill a long experiment run
                self.error = f"{type(e).__name__}: {e}"
                self.outcome = "error"

    def result(self, buyer: Agent, seller: Agent, config_hash: str = "") -> GameResult:
        return GameResult(
            scenario=self.scenario,
            seller_agent=seller.name,
            buyer_agent=buyer.name,
            seller_model=seller.model,
            buyer_model=buyer.model,
            turns=self.turns,
            outcome=self.outcome or "error",
            deal_price=self.deal_price,
            config_hash=config_hash,
            error=self.error,
        )
