"""Agent A: a single system prompt, no planning and no tools. The baseline for RQ1."""

from haggle.agents.base import Observation
from haggle.agents.llm_agent import PromptedAgent, money


class PromptedSeller(PromptedAgent):
    prompt_file = "seller_a.md"

    def __init__(self, client, name: str = "prompted_seller"):
        super().__init__(client, name)

    def prompt_vars(self, obs: Observation) -> dict:
        listing = obs.listing
        return {
            "title": listing.title,
            "category": listing.category,
            "description": listing.description,
            "condition_facts": "\n".join(f"- {fact}" for fact in listing.condition_facts),
            "asking_price": money(listing.asking_price),
            "floor": money(obs.private_limit),
        }
