import pytest

from haggle.scenarios import load_listings, make_scenario, make_scenarios
from haggle.schemas import Listing

LISTING = Listing(id="x", title="t", category="c", description="", condition_facts=["f"], asking_price=200)


def test_same_seed_same_scenario():
    assert make_scenario(LISTING, 7) == make_scenario(LISTING, 7)


def test_different_seeds_differ():
    assert len({(s.seller_floor, s.buyer_budget) for s in (make_scenario(LISTING, i) for i in range(20))}) > 1


@pytest.mark.parametrize("asking", [1, 2, 5, 50, 650, 5000])
def test_budget_always_above_floor_and_floor_in_range(asking):
    lst = LISTING.model_copy(update={"asking_price": asking})
    for seed in range(1000):
        sc = make_scenario(lst, seed)
        assert sc.buyer_budget > sc.seller_floor
        # rounding to whole dollars can move the floor by up to $0.5 either way
        assert asking * 0.70 - 0.5 <= sc.seller_floor <= asking * 0.90 + 0.5


def test_budget_within_brief_range():
    for seed in range(1000):
        sc = make_scenario(LISTING, seed)
        spread = LISTING.asking_price - sc.seller_floor
        assert sc.buyer_budget <= sc.seller_floor + spread * 1.1 + 1  # +1 for rounding/clamp


def test_make_scenarios_stable_order_and_unique_ids():
    lst2 = LISTING.model_copy(update={"id": "y"})
    a = make_scenarios([LISTING, lst2], [0, 1], ["fair", "tough"])
    b = make_scenarios([LISTING, lst2], [0, 1], ["fair", "tough"])
    assert a == b and len(a) == 8
    assert len({s.id for s in a}) == 8
    assert [s.id for s in a][:2] == ["x-s0-fair", "x-s0-tough"]


def test_personas_share_private_values():
    fair, tough = make_scenarios([LISTING], [3], ["fair", "tough"])
    assert (fair.seller_floor, fair.buyer_budget) == (tough.seller_floor, tough.buyer_budget)


def test_load_listings_validates():
    assert load_listings("data/listings_sg.json")
