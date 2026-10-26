"""Deterministic tools for agents. `call_tool` returns a dict ready for `Turn.tool_calls`."""

from dataclasses import dataclass
from typing import Callable

from haggle.tools.calculator import OPS, calculate
from haggle.tools.price_lookup import price_lookup
from haggle.tools.walk_away import walk_away_value


@dataclass
class Tool:
    fn: Callable
    description: str
    parameters: dict  # JSON schema, usable directly as a function-calling spec


TOOLS: dict[str, Tool] = {
    "price_lookup": Tool(
        fn=price_lookup,
        description="Median sold price and 3 closest comparables for an item.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Item name, e.g. 'iPhone 13 128GB'"},
                "category": {"type": "string", "description": "Optional category filter"},
            },
            "required": ["query"],
        },
    ),
    "calculator": Tool(
        fn=calculate,
        description="Price arithmetic: pct_off, midpoint, margin_over_floor, clamp, next_concession.",
        parameters={
            "type": "object",
            "properties": {"op": {"type": "string", "enum": sorted(OPS)}},
            "required": ["op"],
            "additionalProperties": True,
        },
    ),
    "walk_away_value": Tool(
        fn=walk_away_value,
        description="Estimate the final price if the seller keeps negotiating, and whether to accept now.",
        parameters={
            "type": "object",
            "properties": {
                "floor": {"type": "number"},
                "buyer_offers": {"type": "array", "items": {"type": "number"}},
                "market_median": {"type": ["number", "null"]},
                "turns_left": {"type": "integer"},
                "days_remaining": {"type": ["number", "null"], "description": "Optional days left before deadline."},
                "highest_competing_offer": {"type": ["number", "null"], "description": "Optional top outside offer."},
            },
            "required": ["floor", "buyer_offers", "market_median", "turns_left"],
        },
    ),
}


def call_tool(name: str, args: dict) -> dict:
    """Run a tool and return {"name", "args", "result"}. Bad calls give {"error": ...}
    in `result` instead of raising, so a confused model can't crash a game."""
    tool = TOOLS.get(name)
    if tool is None:
        return {"name": name, "args": args, "result": {"error": f"unknown tool '{name}'"}}
    try:
        return {"name": name, "args": args, "result": tool.fn(**args)}
    except (TypeError, ValueError, KeyError) as e:
        return {"name": name, "args": args, "result": {"error": f"{type(e).__name__}: {e}"}}
