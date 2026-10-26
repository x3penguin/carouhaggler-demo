"""Listing loading and seeded floor/budget generation (brief §9)."""

import json
import random
from pathlib import Path

from haggle.schemas import Listing, Scenario


def load_listings(path: str | Path) -> list[Listing]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Listing.model_validate(item) for item in data]


def make_scenario(
    listing: Listing,
    seed: int,
    buyer_persona: str = "human",
    seller_deadline_days: float = 7.0,
    buyer_deadline_days: float = 5.0,
    deadline_days: float | None = None,
) -> Scenario:
    """Draw a private floor and budget for one listing.

    Seeded per (listing, seed) so the same scenario can be replayed against
    different agents. Values are whole dollars because real sellers think in round numbers.
    """
    rng = random.Random(f"{listing.id}:{seed}")
    asking = listing.asking_price
    floor = round(asking * rng.uniform(0.70, 0.90))
    budget = round(floor + (asking - floor) * rng.uniform(0.3, 1.1))
    budget = max(budget, floor + 1)  # a deal must always be possible
    s_days = deadline_days if deadline_days is not None else seller_deadline_days
    b_days = deadline_days if deadline_days is not None else buyer_deadline_days
    return Scenario(
        id=f"{listing.id}-s{seed}",
        listing=listing,
        seller_floor=floor,
        buyer_budget=budget,
        buyer_persona=buyer_persona,
        seed=seed,
        seller_deadline_days=s_days,
        buyer_deadline_days=b_days,
    )


def make_scenarios(
    listings: list[Listing], seeds: list[int], personas: list[str]
) -> list[Scenario]:
    """Deterministic cross-product of listings x seeds x personas, in stable order.

    Floor and budget depend only on (listing, seed), so every persona and every agent
    faces identical private values and results are directly comparable. Ids get a persona
    suffix so they stay unique across the batch.
    """
    out = []
    for listing in listings:
        for seed in seeds:
            for persona in personas:
                sc = make_scenario(listing, seed, persona)
                out.append(sc.model_copy(update={"id": f"{sc.id}-{persona}"}))
    return out
