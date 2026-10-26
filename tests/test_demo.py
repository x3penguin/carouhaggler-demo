"""Test that app/demo.py runs, renders, and handles user interactions without exceptions."""

from pathlib import Path
from streamlit.testing.v1 import AppTest

from haggle.agents.sanity import AcceptAboveFloorSeller

DEMO_PATH = str(Path(__file__).resolve().parents[1] / "app" / "demo.py")


def test_demo_app_initial_render():
    at = AppTest.from_file(DEMO_PATH)
    at.run()
    assert not at.exception, [f"{e.message}\n{e.value}" for e in at.exception]


def test_demo_app_ghost_action():
    at = AppTest.from_file(DEMO_PATH)
    at.run()
    # Mock seller to rule-based sanity agent to avoid live model API latency
    at.session_state.seller = AcceptAboveFloorSeller()
    for b in at.button:
        if "Ghost" in getattr(b, "label", ""):
            b.click()
            at.run()
            break
    assert not at.exception, [f"{e.message}\n{e.value}" for e in at.exception]


def test_demo_app_multi_buyer_run():
    at = AppTest.from_file(DEMO_PATH)
    at.run()
    # Click the tournament button in tab 2
    for b in at.button:
        if "Run Multi-Buyer Tournament" in getattr(b, "label", ""):
            b.click()
            at.run()
            break
    assert not at.exception, [f"{e.message}\n{e.value}" for e in at.exception]
