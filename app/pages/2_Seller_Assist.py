"""RQ6 human arm: review an LLM seller's drafts against an LLM buyer, on the autopilot arm's scenarios.

Opens as a page of the demo. Games are saved under runs/assist_human/<timestamp>/. The results notebook pools every session
there and compares them with the autopilot arm: `uv run python -m haggle.runner configs/assist.yaml`.
"""

import json
from datetime import datetime

import streamlit as st

from haggle import assist
from haggle.agents.llm_agent import money
from haggle.config import PROJECT_ROOT, check_models, config_hash, load_config
from haggle.engine import Game
from haggle.llm import CostTracker
from haggle.runner import BUYERS, SELLERS, BuildContext, build_agent, build_jobs
from haggle.schemas import Action

CONFIG_PATH = "configs/assist.yaml"
ACTIONS = ["offer", "chat", "accept", "reject", "walk_away"]

st.set_page_config(page_title="Haggle Arena · Seller Assist", page_icon="✍️", layout="wide")
st.header("✍️ Seller Assist")
st.caption("The seller agent drafts each reply. Approve it, or edit it before it goes to the buyer.")

cfg = load_config(CONFIG_PATH)
jobs = build_jobs(cfg)
ss = st.session_state
try:
    check_models(cfg, {jobs[0].seller["model"], jobs[0].buyer["model"]})
except ValueError as e:
    st.error(str(e))
    st.stop()


@st.cache_resource
def session_dir():
    d = PROJECT_ROOT / "runs" / "assist_human" / datetime.now().strftime("%Y%m%d-%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.yaml").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return d


@st.cache_resource
def tracker():
    return CostTracker(float(cfg["max_budget_usd"]))


def start(job) -> None:
    scenario = job.scenario()
    log = session_dir() / "llm_calls.jsonl"
    ctx = BuildContext(cfg, scenario, log, tracker())
    ss.assist_game = Game(
        scenario,
        max_turns_per_side=cfg["engine"]["max_turns_per_side"],
        max_parse_retries=cfg["engine"]["max_parse_retries"],
        floor_protection=job.floor_protection,
        budget_protection=True,
    )
    ss.assist_seller = assist.ReviewedSeller(build_agent(SELLERS, job.seller, ctx))
    ss.assist_buyer = build_agent(BUYERS, job.buyer, ctx)
    ss.assist_draft, ss.assist_saved, ss.assist_error = None, False, None


def save() -> None:
    game = ss.assist_game
    result = game.result(ss.assist_buyer, ss.assist_seller, config_hash(cfg))
    out = session_dir()
    with (out / "games.jsonl").open("a", encoding="utf-8") as f:
        f.write(result.model_dump_json() + "\n")
    played = sum(1 for line in (out / "games.jsonl").open(encoding="utf-8") if line.strip())
    spent = tracker().spent_usd
    # The Results page lists runs by their manifest.
    (out / "manifest.json").write_text(json.dumps({
        "experiment": "assist_human", "config_hash": config_hash(cfg), "games_planned": len(jobs),
        "games_played": played, "error_games": 0, "spent_usd": spent, "cost_per_game_usd": spent / played,
        "budget_usd": float(cfg["max_budget_usd"]), "aborted": None,
    }, indent=2), encoding="utf-8")
    ss.assist_saved = True


def run_model(fn) -> None:
    try:
        fn()
    except Exception as e:  # noqa: BLE001  keep the session alive so the reviewer can retry
        ss.assist_error = f"The model call failed ({type(e).__name__}: {e}). Press Retry."


def advance() -> None:
    """Let the buyer move if it's its turn, then fetch the seller's next draft."""
    game = ss.assist_game
    if not game.is_over and game.next_speaker == "buyer":
        game.step(ss.assist_buyer)
    if not game.is_over and game.next_speaker == "seller" and ss.assist_draft is None:
        ss.assist_draft = assist.draft(game, ss.assist_seller)


with st.sidebar:
    idx = st.selectbox("Scenario", range(len(jobs)), format_func=lambda i: f"{i + 1}. {jobs[i].listing.title}")
    if st.button("▶️ Start this scenario", use_container_width=True, type="primary"):
        start(jobs[idx])
        with st.spinner("Waiting for the buyer and the seller's draft..."):
            run_model(advance)
        st.rerun()

if "assist_game" not in ss:
    st.info("Choose a scenario and press Start. Play each scenario once.")
    st.stop()

game: Game = ss.assist_game
scenario = game.scenario
listing = scenario.listing
left, right = st.columns([3, 2], gap="large")

with right, st.container(border=True):
    st.subheader("Your seller brief")
    st.metric("Asking price", f"SGD {money(listing.asking_price)}")
    st.metric("Your private floor", f"SGD {money(scenario.seller_floor)}")
    st.markdown("**True condition of the item**")
    for fact in listing.condition_facts:
        st.markdown(f"- {fact}")

with left:
    with st.container(border=True):
        st.subheader(listing.title)
        st.write(listing.description)
    for t in game.turns:
        with st.chat_message("user" if t.speaker == "buyer" else "assistant", avatar="🧑" if t.speaker == "buyer" else "🏷️"):
            st.write(t.action.message or "_(no message)_")
            st.caption(t.action.action + (f" SGD {money(t.action.price)}" if t.action.action == "offer" else ""))

    if ss.assist_error:
        st.error(ss.assist_error)

    if game.is_over:
        if not ss.assist_saved:
            save()
        price = f" at SGD {money(game.deal_price)}" if game.deal_price is not None else ""
        st.success(f"Game over: {game.outcome}{price}. Saved. Choose the next scenario in the sidebar.")
    elif ss.assist_draft is None:
        if st.button("Retry"):
            ss.assist_error = None
            with st.spinner("Waiting for the model..."):
                run_model(advance)
            st.rerun()
    else:
        decision = ss.assist_draft
        d = decision.action
        with st.form(f"review_{len(game.turns)}"):
            st.subheader("Draft reply")
            message = st.text_area("Message to buyer", value=d.message, height=110)
            c1, c2, c3 = st.columns(3)
            action = c1.selectbox("Action", ACTIONS, index=ACTIONS.index(d.action))
            price = c2.number_input("Price (offers only)", min_value=0.0, step=1.0, value=float(d.price or 0.0))
            wait = c3.number_input("Wait (hours)", min_value=0, max_value=72, step=1, value=d.wait_hours)
            approve = st.form_submit_button("✅ Approve draft as written", use_container_width=True)
            send_edit = st.form_submit_button("✏️ Send my edited version", use_container_width=True)
        if approve or send_edit:
            edited = None
            if send_edit:
                edited = Action(message=message, action=action, price=price if action == "offer" else None, wait_hours=int(wait))
            try:
                assist.submit_review(game, decision, edited)
            except ValueError as e:
                st.error(str(e))
                st.stop()
            ss.assist_draft, ss.assist_error = None, None
            with st.spinner("Waiting for the buyer..."):
                run_model(advance)
            st.rerun()
