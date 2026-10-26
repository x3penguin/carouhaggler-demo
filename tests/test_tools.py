import json

import pytest

from haggle.schemas import Turn
from haggle.tools import TOOLS, call_tool
from haggle.tools.calculator import calculate, clamp, margin_over_floor, midpoint, next_concession, pct_off
from haggle.tools.price_lookup import price_lookup
from haggle.tools.walk_away import walk_away_value


@pytest.fixture()
def comps(tmp_path):
    rows = [
        ("a", "phones", "iPhone 13 128GB Midnight", 600),
        ("b", "phones", "iPhone 13 128GB Blue", 640),
        ("c", "phones", "iPhone 13 256GB", 700),
        ("d", "phones", "Samsung Galaxy S22", 400),
        ("e", "laptops", "iPhone case laptop sleeve", 20),
    ]
    data = [dict(id=i, category=c, title=t, condition="good", price_sgd=p, age_months=12, source="t", notes="") for i, c, t, p in rows]
    path = tmp_path / "c.json"
    path.write_text(json.dumps(data))
    return path


# ---- price_lookup ------------------------------------------------------------


def test_lookup_returns_expected_median_and_examples(comps):
    r = price_lookup("iPhone 13 128GB", "phones", path=comps)
    assert r["n"] == 3 and r["median"] == 640
    assert [e["id"] for e in r["examples"]][:2] == ["a", "b"]
    assert len(r["examples"]) == 3


def test_category_filter_excludes_other_categories(comps):
    r = price_lookup("iPhone", "phones", path=comps)
    assert all(e["id"] != "e" for e in r["examples"])
    assert price_lookup("iPhone", None, path=comps)["n"] == 4


def test_no_match_returns_empty_result(comps):
    assert price_lookup("spaceship", path=comps) == {"median": None, "n": 0, "examples": []}


def test_lookup_is_deterministic(comps):
    assert price_lookup("iPhone 13", path=comps) == price_lookup("iPhone 13", path=comps)


def test_real_comparables_file_loads():
    assert price_lookup("iPhone 13 128GB")["median"] is not None


# ---- calculator --------------------------------------------------------------


def test_calculator_values():
    assert pct_off(200, 150) == pytest.approx(25.0)
    assert pct_off(100, 120) == pytest.approx(-20.0)
    assert midpoint(80, 100) == 90
    assert margin_over_floor(95, 80) == 15 and margin_over_floor(70, 80) == -10
    assert clamp(50, 60, 90) == 60 and clamp(100, 60, 90) == 90 and clamp(75, 60, 90) == 75


def test_calculator_edge_cases():
    with pytest.raises(ValueError):
        pct_off(0, 10)
    with pytest.raises(ValueError):
        clamp(5, 10, 1)
    assert pct_off(100, 100) == 0


def test_next_concession():
    assert next_concession(100, 80, 4) == pytest.approx(95)
    assert next_concession(100, 80, 1) == 80
    assert next_concession(80, 80, 5) == 80 and next_concession(70, 80, 5) == 80
    assert calculate("midpoint", a=1, b=3) == 2
    with pytest.raises(ValueError):
        calculate("nope")


# ---- walk-away ---------------------------------------------------------------


def test_walk_away_no_offers():
    r = walk_away_value(80, [], 100, 5)
    assert r["best_offer"] is None and not r["accept_now_advised"]


def test_walk_away_below_floor_keeps_going():
    r = walk_away_value(80, [50, 60], 100, 4)
    assert not r["accept_now_advised"] and "below the floor" in r["reason"]


def test_walk_away_accepts_when_no_more_to_gain():
    r = walk_away_value(80, [85, 85, 85], 90, 4)  # flat trend: nothing more coming
    assert r["accept_now_advised"] and r["expected_final"] == 85


def test_walk_away_caps_at_market_median():
    r = walk_away_value(80, [60, 70, 80], 90, 10)  # trend would project 180
    assert r["expected_final"] == 90


def test_walk_away_monotonic_in_buyer_offers():
    base = [60, 65, 70]
    finals = [walk_away_value(80, [x + d for x in base], 120, 3)["expected_final"] for d in (0, 5, 10, 20)]
    assert finals == sorted(finals)


# ---- registry ----------------------------------------------------------------


def test_call_tool_matches_turn_tool_calls_format():
    rec = call_tool("calculator", {"op": "midpoint", "a": 2, "b": 4})
    assert rec == {"name": "calculator", "args": {"op": "midpoint", "a": 2, "b": 4}, "result": 3}
    from haggle.schemas import Action
    Turn(index=0, speaker="seller", action=Action(message="", action="reject", price=None), tool_calls=[rec])


def test_call_tool_never_raises_on_bad_calls():
    assert "error" in call_tool("nope", {})["result"]
    assert "error" in call_tool("calculator", {"op": "pct_off", "asking": 0, "price": 1})["result"]
    assert "error" in call_tool("price_lookup", {})["result"]


def test_registry_has_schemas():
    assert set(TOOLS) == {"price_lookup", "calculator", "walk_away_value"}
    assert all(t.parameters["type"] == "object" for t in TOOLS.values())
