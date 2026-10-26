"""Price arithmetic helpers. Plain functions with explicit arguments, no eval()."""


def pct_off(asking: float, price: float) -> float:
    """Percentage below the asking price (negative if the price is above asking)."""
    if asking <= 0:
        raise ValueError("asking must be positive")
    return (asking - price) / asking * 100


def midpoint(a: float, b: float) -> float:
    return (a + b) / 2


def margin_over_floor(price: float, floor: float) -> float:
    """Dollars above the floor (negative means below it)."""
    return price - floor


def clamp(price: float, lo: float, hi: float) -> float:
    if lo > hi:
        raise ValueError("lo must not exceed hi")
    return max(lo, min(hi, price))


def next_concession(current: float, floor: float, remaining_turns: int) -> float:
    """Next lower asking price if the remaining gap to the floor is conceded evenly.

    remaining_turns counts this turn. On the last turn the result is the floor itself.
    Never returns below the floor.
    """
    if current <= floor:
        return floor
    if remaining_turns <= 1:
        return floor
    return max(floor, current - (current - floor) / remaining_turns)


OPS = {
    "pct_off": pct_off,
    "midpoint": midpoint,
    "margin_over_floor": margin_over_floor,
    "clamp": clamp,
    "next_concession": next_concession,
}


def calculate(op: str, **args) -> float:
    """Dispatch to one of OPS by name. Used by the tool registry."""
    if op not in OPS:
        raise ValueError(f"unknown op '{op}', choose from {sorted(OPS)}")
    return OPS[op](**args)
