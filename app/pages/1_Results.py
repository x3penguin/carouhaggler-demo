"""Results browser: pick a run directory and see its summary table with 95% CIs, plus single games.

Opens as a second page of the demo (Streamlit multipage): uv run streamlit run app/demo.py
"""

import json

import streamlit as st

from haggle.config import PROJECT_ROOT
from haggle.metrics import game_metrics, load_honesty, load_results, game_id, summarize

st.set_page_config(page_title="Haggle Arena · Results", page_icon="📊", layout="wide")
st.header("📊 Experiment results")

runs = sorted((p.parent for p in (PROJECT_ROOT / "runs").rglob("manifest.json")), reverse=True)
if not runs:
    st.info("No experiment runs yet. Try `uv run python -m haggle.runner configs/sanity.yaml`.")
    st.stop()

run = st.selectbox("Run", runs, format_func=lambda p: str(p.relative_to(PROJECT_ROOT / "runs")))
manifest = json.loads((run / "manifest.json").read_text())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Games", f"{manifest['games_played']} / {manifest['games_planned']}")
c2.metric("Spent", f"${manifest['spent_usd']:.4f}")
c3.metric("Cost per game", "—" if manifest["cost_per_game_usd"] is None else f"${manifest['cost_per_game_usd']:.4f}")
c4.metric("Budget", f"${manifest['budget_usd']:.2f}")
if manifest["aborted"]:
    st.warning(f"Run stopped early: {manifest['aborted']}")

results = load_results(run)
verdicts = load_honesty(run / "honesty.jsonl")
rows = [game_metrics(r, verdicts.get(game_id(r)) if verdicts.get(game_id(r), {}).get("violations") is not None else None)
        for r in results]

group_options = ["seller_agent", "seller_model", "buyer_persona", "buyer_model", "attack_category", "attack_kind", "category"]
group_by = st.multiselect("Group by", group_options, default=["seller_agent", "buyer_persona"])
METRICS = {
    "deal_rate": True, "surplus_share": False, "seller_payoff": False, "below_floor_rate": True, "floor_leak_rate": True,
    "attack_success_rate": True, "dishonest_rate": True, "turns_to_deal": False, "cost_per_game_usd": False,
}


def cell(s, m, pct):
    v, lo, hi = s[m], s[f"{m}_lo"], s[f"{m}_hi"]
    if v is None:
        return "—"
    f = (lambda x: f"{100 * x:.0f}%") if pct else (lambda x: f"{x:.3f}")
    return f(v) if lo == hi else f"{f(v)} [{f(lo)}, {f(hi)}]"


if group_by:
    table = [{**{k: s[k] for k in group_by}, "games": s["games"], **{m: cell(s, m, p) for m, p in METRICS.items()}}
             for s in summarize(rows, group_by=tuple(group_by))]
    st.caption("Mean [95% bootstrap CI] over games.")
    st.dataframe(table, use_container_width=True, hide_index=True)

st.subheader("Inspect a game")
idx = st.selectbox("Game", range(len(results)),
                   format_func=lambda i: f"{results[i].scenario.id} · {results[i].seller_agent} vs {results[i].buyer_agent} · {results[i].outcome}")
r, m = results[idx], rows[idx]
left, right = st.columns([3, 2])
with left:
    for t in r.turns:
        with st.chat_message("user" if t.speaker == "buyer" else "assistant", avatar="🧑" if t.speaker == "buyer" else "🏷️"):
            st.write(t.action.message or "_(no message)_")
            st.caption(f"{t.action.action}{' SGD ' + str(t.action.price) if t.action.action == 'offer' else ''}")
            for e in t.guardrail_events:
                st.warning(e, icon="🛡️")
with right:
    st.metric("Floor / budget", f"${r.scenario.seller_floor:.0f} / ${r.scenario.buyer_budget:.0f}")
    st.json({k: m[k] for k in ["outcome", "deal_price", "surplus_share", "below_floor", "floor_leak",
                               "seller_backward_offers", "attack_success", "honesty_violations", "cost_usd"]})
    if game_id(r) in verdicts:
        st.markdown("**Judge verdict**")
        st.json(verdicts[game_id(r)])
