"""Walk-away value estimator: is it worth continuing, or should the seller take the offer?

Deterministic and transparent on purpose, so Agent B can cite it and a human can check it.
It extrapolates the buyer's own concession trend, caps the projection at what the market
supports, and compares the result with the best offer already on the table.
"""


def walk_away_value(
    floor: float,
    buyer_offers: list[float],
    market_median: float | None,
    turns_left: int,
    days_remaining: float | None = None,
    highest_competing_offer: float | None = None,
) -> dict:
    """Estimate where the negotiation ends up if the seller keeps going.

    buyer_offers: the buyer's offers in order. turns_left: seller turns remaining.
    days_remaining: optional simulated days left before deadline.
    highest_competing_offer: optional outside bid from another buyer.
    Returns {"expected_final", "best_offer", "accept_now_advised", "reason"}.
    """
    effective_floor = max(floor, highest_competing_offer) if highest_competing_offer is not None else floor

    if not buyer_offers:
        expected = market_median if market_median is not None else effective_floor
        return {
            "expected_final": float(max(expected, effective_floor)),
            "best_offer": None,
            "accept_now_advised": False,
            "reason": "No buyer offer yet, so there is nothing to accept.",
        }

    best = max(buyer_offers)
    last = buyer_offers[-1]
    n = len(buyer_offers)
    trend = (last - buyer_offers[0]) / (n - 1) if n >= 2 else 0.0
    projected = last + max(trend, 0.0) * max(turns_left, 0)
    # The market caps what a buyer should pay, but never below what they already offered.
    ceiling = max(market_median, last) if market_median is not None else projected
    expected = min(projected, ceiling)

    # Deadline pressure: if under 1 day remaining, threshold loosens
    threshold = 1.05 if (days_remaining is not None and days_remaining < 1.0) else 1.02
    accept = best >= effective_floor and expected <= best * threshold

    if highest_competing_offer is not None and best < highest_competing_offer:
        reason = f"A competing buyer offered ${highest_competing_offer:.0f}; do not accept ${best:.0f}."
    elif best < floor:
        reason = f"Best offer ${best:.0f} is below the floor ${floor:.0f}; keep negotiating."
    elif accept:
        reason = f"Expected final ${expected:.0f} is barely above the best offer ${best:.0f}; take it."
    else:
        reason = f"Expected final ${expected:.0f} beats the best offer ${best:.0f}; keep negotiating."
    return {
        "expected_final": float(expected),
        "best_offer": float(best),
        "accept_now_advised": bool(accept),
        "reason": reason,
    }
