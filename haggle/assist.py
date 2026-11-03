"""RQ6 assist mode: the seller drafts each reply, and a human approves or edits it.

Built on the engine's decide/commit split, so a draft goes through the normal retries and is
only delivered after review. Each decision is logged as a guardrail event and an edited
draft is kept in private_notes, so no schema change is needed.

The reviewer gets no warnings about the draft: the comparison is human judgement against
autopilot, and automatic warnings would measure the warning instead.
"""

from haggle.agents.base import Agent
from haggle.engine import Decision, Game
from haggle.metrics import ASSIST_APPROVED, ASSIST_EDITED
from haggle.schemas import Action, Turn


class ReviewedSeller:
    """Same seller under a new name, so assist games stay separate from autopilot in every table."""

    def __init__(self, seller: Agent):
        self.seller = seller
        self.name = f"{seller.name}+human_review"
        self.model = seller.model

    def act(self, obs):
        return self.seller.act(obs)


def draft(game: Game, seller: Agent) -> Decision:
    """The seller's proposed reply, not yet delivered."""
    if game.next_speaker != "seller":
        raise ValueError("It's the buyer's turn, so there's nothing to draft.")
    return game.decide(seller)


def submit_review(game: Game, decision: Decision, edited: Action | None = None) -> Turn:
    """Deliver the draft, or the human's edit of it. Raises ValueError on an illegal move."""
    final = edited if edited is not None else decision.action
    error = game.rule_error("seller", final)
    if error is None and game.guardrail_violation("seller", final):
        error = "The floor guardrail blocks that price."
    if error:
        raise ValueError(error)
    changed = final != decision.action
    if changed:
        original = f"[draft before human edit] {decision.action.model_dump_json()}"
        decision.private_notes = "\n".join(filter(None, [decision.private_notes, original]))
    decision.action = final
    decision.guardrail_events = [*decision.guardrail_events, ASSIST_EDITED if changed else ASSIST_APPROVED]
    return game.commit(decision)
