"""LLM buyer personas. The persona only changes the style text; the budget is enforced by the engine."""

from haggle.agents.base import Observation
from haggle.agents.llm_agent import PromptedAgent, money

PERSONAS = {
    "fair": "You negotiate politely and reasonably. You value finding middle ground and closing a good deal without unnecessary friction.",
    "tough": "You are an assertive bargainer. You scrutinize the item's flaws, concede ground very cautiously, and firmly test how low the seller can go.",
    "lowballer": "You are an aggressive bargain hunter looking for steep discounts. You test sellers with bold low offers, speak in brief casual local phrasing, and give concessions very reluctantly.",
}


class LLMBuyer(PromptedAgent):
    prompt_file = "buyer.md"

    def __init__(self, client, persona: str = "fair"):
        super().__init__(client, name=f"llm_buyer:{persona}")
        self.persona = persona

    def prompt_vars(self, obs: Observation) -> dict:
        listing = obs.listing
        target_days = f"{obs.deadline_hours / 24.0:.1f}" if obs.deadline_hours else "5.0"
        return {
            "title": listing.title,
            "category": listing.category,
            "description": listing.description,
            "asking_price": money(listing.asking_price),
            "budget": money(obs.private_limit),
            "persona": PERSONAS[self.persona],
            "target_days": target_days,
        }
