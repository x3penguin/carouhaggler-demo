"""LLM buyer personas. The persona only changes the style text; the budget is enforced by the engine."""

from haggle.agents.base import Observation
from haggle.agents.llm_agent import PromptedAgent, money

PERSONAS = {
    "fair": "You negotiate politely and reasonably. Open somewhat below the asking price and "
    "meet the seller partway. You are happy to close a fair deal quickly.",
    "tough": "You are a hard bargainer. Open well below asking, concede in small steps, point out "
    "flaws in the item, and threaten to walk away if the seller won't move.",
    "lowballer": "You always start with a very low offer, about half the asking price, and only "
    "raise reluctantly. Short messages, local style, e.g. 'can nego?', 'lowest?', 'meet at MRT?'.",
}


class LLMBuyer(PromptedAgent):
    prompt_file = "buyer.md"

    def __init__(self, client, persona: str = "fair"):
        super().__init__(client, name=f"llm_buyer:{persona}")
        self.persona = persona

    def prompt_vars(self, obs: Observation) -> dict:
        listing = obs.listing
        return {
            "title": listing.title,
            "category": listing.category,
            "description": listing.description,
            "asking_price": money(listing.asking_price),
            "budget": money(obs.private_limit),
            "persona": PERSONAS[self.persona],
        }
