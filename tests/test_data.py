import json
from pathlib import Path

from haggle.scenarios import load_listings

PATH = Path("data/listings_sg.json")


def test_at_least_100_listings():
    assert len(load_listings(PATH)) >= 100


def test_ids_unique_and_prices_positive():
    listings = load_listings(PATH)
    assert len({lst.id for lst in listings}) == len(listings)
    assert all(lst.asking_price > 0 for lst in listings)


def test_every_listing_has_a_flaw_and_facts():
    # The brief needs at least one flaw so agents have something to lie about (RQ4).
    for lst in load_listings(PATH):
        assert len(lst.condition_facts) >= 2, lst.id


def test_json_is_a_list_of_objects_with_expected_keys():
    raw = json.loads(PATH.read_text(encoding="utf-8"))
    assert all(set(r) == {"id", "title", "category", "description", "condition_facts", "asking_price"} for r in raw)
