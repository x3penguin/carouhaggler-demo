"""Game engine: turn order, legal moves, guardrails and outcome resolution (brief §6).

All negotiation rules live here, in code, so a model can never "decide" that a deal
happened. Agents only propose actions; the engine decides what they mean.
"""

from haggle.agents.base import Agent, Observation, PublicTurn
from haggle.schemas import Action, GameResult, Scenario, Side, Turn


def other(side: Side) -> Side:
    return "seller" if side == "buyer" else "buyer"


class Game:
    def __init__(
        self,
        scenario: Scenario,
        max_turns_per_side: int = 10,
        max_parse_retries: int = 2,
        floor_protection: bool = True,
        budget_protection: bool = True,
    ):
        self.scenario = scenario
        self.max_turns_per_side = max_turns_per_side
        self.max_parse_retries = max_parse_retries
        # Switchable because baseline runs must be able to *show* the failure (§6.8).
        self.floor_protection = floor_protection
        # Stops LLM buyers paying over budget. Off for human players, who may do as they like.
        self.budget_protection = budget_protection

        self.turns: list[Turn] = []
        # The asking price is the seller's opening offer (§6.5).
        self.standing: dict[Side, float | None] = {
            "buyer": None,
            "seller": scenario.listing.asking_price,
        }
        self.outcome: str | None = None
        self.deal_price: float | None = None

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

    def observation(self, side: Side, feedback: str | None = None) -> Observation:
        """Everything `side` is allowed to see: public transcript + its own limit only (§6.6)."""
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
                )
                for t in self.turns
            ],
            my_standing_offer=self.standing[side],
            their_standing_offer=self.standing[other(side)],
            turns_left=self.turns_left(side),
            feedback=feedback,
        )

    # ---- validation --------------------------------------------------------

    def rule_error(self, side: Side, action: Action) -> str | None:
        """Moves that are illegal under any settings. Returns a message for the agent, or None."""
        if action.action == "offer" and (action.price is None or action.price <= 0):
            return "An 'offer' needs a positive price."
        if action.action == "accept" and self.standing[other(side)] is None:
            return f"The {other(side)} has not made an offer yet, so there is nothing to accept."
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
        """Tell the agent exactly why it was blocked and what a legal alternative looks like."""
        limit = self.private_limit(side)
        if side == "seller":
            return (
                f"BLOCKED. You may not sell below your floor of ${limit:.0f}. "
                f"Do not accept. Make a counter-offer of at least ${limit:.0f} instead."
            )
        return (
            f"BLOCKED. You may not pay more than your budget of ${limit:.0f}. "
            f"Do not accept. Make a counter-offer of at most ${limit:.0f} instead."
        )

    # ---- the turn loop -----------------------------------------------------

    def step(self, agent: Agent) -> Turn:
        """Ask the next speaker for an action, enforce the rules, and apply it."""
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

        turn = Turn(
            index=len(self.turns),
            speaker=side,
            action=action,
            private_notes=notes,
            tool_calls=tool_calls,
            parse_retries=retries,
            guardrail_events=events,
            **usage,
        )
        self.turns.append(turn)
        self._apply(side, action)
        return turn

    def _apply(self, side: Side, action: Action) -> None:
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

    def play(self, buyer: Agent, seller: Agent) -> None:
        """Run to completion with two automated agents."""
        while not self.is_over:
            self.step(buyer if self.next_speaker == "buyer" else seller)

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
        )
