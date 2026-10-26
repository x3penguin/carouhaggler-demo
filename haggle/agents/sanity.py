"""Rule-based sanity agents with known outcomes (brief §7).

TEST-ONLY. Several of these are constructed with the Scenario and therefore know the
opponent's private floor or budget, which a real agent never may (brief §6.6). They exist
so the engine and metrics can be validated against exact expected values before any LLM
result is trusted. Never use them as experiment baselines.
"""

from haggle.agents.base import AgentReply, Observation
from haggle.schemas import Action, Scenario

ACCEPT = Action(message="Deal.", action="accept", price=None)
REJECT = Action(message="No.", action="reject", price=None)


def _offer(price: float, message: str = "") -> Action:
    return Action(message=message or f"How about ${price:.0f}?", action="offer", price=price)


class SanityAgent:
    """Common bookkeeping: a name and a model label so results can be logged like any agent."""

    model = "rule_based"

    def __init__(self, name: str):
        self.name = name

    def decide(self, obs: Observation) -> Action:
        raise NotImplementedError

    def act(self, obs: Observation) -> AgentReply:
        return AgentReply(action=self.decide(obs))


# ---- buyers ------------------------------------------------------------------


class FullBudgetBuyer(SanityAgent):
    """Offers its entire budget immediately. Against a seller who takes it, surplus share = 1.0."""

    def __init__(self, scenario: Scenario):
        super().__init__("sanity_full_budget_buyer")
        self.budget = scenario.buyer_budget

    def decide(self, obs: Observation) -> Action:
        if obs.my_standing_offer is None:
            return _offer(self.budget)
        their = obs.their_standing_offer
        if their is not None and their <= self.budget:
            return ACCEPT
        return REJECT


class FloorBuyer(SanityAgent):
    """Offers exactly the seller's floor. Against a seller who takes it, surplus share = 0.0."""

    def __init__(self, scenario: Scenario):
        super().__init__("sanity_floor_buyer")
        self.floor = scenario.seller_floor
        self.budget = scenario.buyer_budget

    def decide(self, obs: Observation) -> Action:
        if obs.my_standing_offer is None:
            return _offer(self.floor)
        their = obs.their_standing_offer
        if their is not None and their <= self.floor:
            return ACCEPT
        return REJECT


class LinearConcessionBuyer(SanityAgent):
    """Raises its offer in equal steps from start_frac * asking up to its budget.

    Gives a predictable turn count: the k-th buyer offer is start + (budget - start) * k / (steps - 1).
    """

    def __init__(self, scenario: Scenario, steps: int = 5, start_frac: float = 0.6):
        super().__init__("sanity_linear_buyer")
        self.budget = scenario.buyer_budget
        self.start = scenario.listing.asking_price * start_frac
        self.steps = steps

    def offer_at(self, k: int) -> float:
        k = min(k, self.steps - 1)
        return self.start + (self.budget - self.start) * k / (self.steps - 1)

    def decide(self, obs: Observation) -> Action:
        k = sum(1 for t in obs.transcript if t.speaker == "buyer")
        planned = self.offer_at(k)
        their = obs.their_standing_offer
        if their is not None and their <= planned:
            return ACCEPT
        return _offer(planned)


# ---- sellers -----------------------------------------------------------------


class AcceptAboveFloorSeller(SanityAgent):
    """Accepts any buyer offer at or above its floor, otherwise holds at the asking price."""

    def __init__(self):
        super().__init__("sanity_accept_above_floor_seller")

    def decide(self, obs: Observation) -> Action:
        their = obs.their_standing_offer
        if their is not None and their >= obs.private_limit:
            return ACCEPT
        return _offer(obs.listing.asking_price)


class StubbornSeller(SanityAgent):
    """Never concedes: re-offers the asking price every turn, so deal rate is about 0."""

    def __init__(self):
        super().__init__("sanity_stubborn_seller")

    def decide(self, obs: Observation) -> Action:
        return _offer(obs.listing.asking_price, "Price is firm.")


class LeakySeller(SanityAgent):
    """Blurts out its floor in a message. Used to check floor-leak detection."""

    def __init__(self):
        super().__init__("sanity_leaky_seller")

    def decide(self, obs: Observation) -> Action:
        return _offer(obs.listing.asking_price, f"Honestly my lowest is ${obs.private_limit:.0f}.")


class BelowFloorSeller(SanityAgent):
    """Accepts any offer regardless of floor. Floor protection must stop it; with it off the
    deal goes through and `below_floor` must flag it."""

    def __init__(self):
        super().__init__("sanity_below_floor_seller")

    def decide(self, obs: Observation) -> Action:
        if obs.their_standing_offer is not None:
            return ACCEPT
        return _offer(obs.listing.asking_price)


class BackwardOfferAgent(SanityAgent):
    """Offers a fixed sequence of prices, one per turn. Default goes down then back up,
    i.e. a 'backward' move, for testing backward-offer detection."""

    def __init__(self, prices: tuple[float, ...] = (90, 80, 85)):
        super().__init__("sanity_backward_offer_agent")
        self.prices = prices

    def decide(self, obs: Observation) -> Action:
        k = sum(1 for t in obs.transcript if t.speaker == obs.role)
        return _offer(self.prices[min(k, len(self.prices) - 1)])


# ---- timed & multi-party sanity agents --------------------------------------


class DelayedBuyer(SanityAgent):
    """Offers its budget with an explicit simulated wait_hours delay."""

    def __init__(self, scenario: Scenario, wait_hours: int = 24):
        super().__init__("sanity_delayed_buyer")
        self.budget = scenario.buyer_budget
        self.wait_hours = wait_hours

    def decide(self, obs: Observation) -> Action:
        their = obs.their_standing_offer
        if their is not None and their <= self.budget:
            return Action(message="Deal.", action="accept", price=None, wait_hours=self.wait_hours)
        return Action(
            message=f"I can offer ${self.budget:.0f}.",
            action="offer",
            price=self.budget,
            wait_hours=self.wait_hours,
        )


class CompetitiveBuyer(SanityAgent):
    """Reacts to outside competing bids: raises offer to outbid rival, or walks away if over budget."""

    def __init__(self, scenario: Scenario, increment: float = 10.0):
        super().__init__("sanity_competitive_buyer")
        self.budget = scenario.buyer_budget
        self.floor = scenario.seller_floor
        self.increment = increment

    def decide(self, obs: Observation) -> Action:
        their = obs.their_standing_offer
        competing = obs.highest_competing_offer

        # If seller accepts our offer or seller's offer is within budget and beats competing
        if their is not None and their <= self.budget:
            if competing is None or their >= competing:
                return ACCEPT

        # Target price: outbid rival if needed
        base = competing if competing is not None else self.floor
        target = base + self.increment
        if target > self.budget:
            return Action(message="Too rich for me, I walk.", action="walk_away", price=None)

        return _offer(target, f"Outbidding rival: ${target:.0f}")


class GhostingSeller(SanityAgent):
    """Stays silent (ghosts with empty message and delay) if offer is below floor; accepts once at/above floor."""

    def __init__(self, scenario: Scenario, wait_hours: int = 12):
        super().__init__("sanity_ghosting_seller")
        self.floor = scenario.seller_floor
        self.wait_hours = wait_hours

    def decide(self, obs: Observation) -> Action:
        their = obs.their_standing_offer
        if their is not None and their >= self.floor:
            return Action(message="Deal.", action="accept", price=None, wait_hours=0)
        return Action(message="", action="reject", price=None, wait_hours=self.wait_hours)


