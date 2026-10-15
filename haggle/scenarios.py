"""Listing loading and seeded floor/budget generation (brief §9)."""

import json
import random
from pathlib import Path

from haggle.schemas import Listing, Scenario


def load_listings(path: str | Path) -> list[Listing]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Listing.model_validate(item) for item in data]


def make_scenario(listing: Listing, seed: int, buyer_persona: str = "human") -> Scenario:
    """Draw a private floor and budget for one listing.

    Seeded per (listing, seed) so the same scenario can be replayed against
    different agents. Values are whole dollars because real sellers think in round numbers.
    """
    rng = random.Random(f"{listing.id}:{seed}")
    asking = listing.asking_price
    floor = round(asking * rng.uniform(0.70, 0.90))
    budget = round(floor + (asking - floor) * rng.uniform(0.3, 1.1))
    budget = max(budget, floor + 1)  # a deal must always be possible
    return Scenario(
        id=f"{listing.id}-s{seed}",
        listing=listing,
        seller_floor=floor,
        buyer_budget=budget,
        buyer_persona=buyer_persona,
        seed=seed,
    )
