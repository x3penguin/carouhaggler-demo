"""Report tables and charts, built from run directories. Used by analysis/results.ipynb."""

from __future__ import annotations

import csv
import math
import json
import re
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

from haggle.harnesses import paired_difference, self_play_calibration
from haggle.judge import agreement, label_id
from haggle.metrics import (
    ATTACK_SUCCESS_RULES,
    SUMMARY_METRICS,
    bootstrap_ci,
    attack_effect,
    floor_leak,
    game_id,
    competing_offer_response,
    concession_by_time,
    game_metrics,
    latest_run,
    load_honesty,
    load_results,
    market_summary,
    summarize,
    surplus_shift,
)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"




def load_rows(*run_dirs: str | Path) -> list[dict]:
    """Metric rows for every game under the given run dirs, joined with judge verdicts."""
    rows = []
    for d in run_dirs:
        d = Path(d)
        verdicts = {}
        for h in d.rglob("honesty.jsonl"):
            verdicts |= load_honesty(h)
        for r in load_results(d):
            if r.outcome == "error":  # a server failure isn't a no-deal; see manifest error_games
                continue
            v = verdicts.get(game_id(r))
            judged = v if v and v.get("violations") is not None else None
            row = game_metrics(r, judged)
            # Major lies only (a flaw denied or a false concrete fact): on 30 hand-labelled games the judge made no
            # false alarms on these, while its "minor" flags were mostly sales talk the labellers accepted.
            row["major_dishonest"] = None if judged is None else (judged.get("major_violations") or 0) > 0
            rows.append(row)
    return rows


def load_games(*run_dirs: str | Path) -> list:
    """Raw GameResults, for the analyses that need each turn rather than per-game rows."""
    return [r for d in run_dirs for r in load_results(d)]


def load_manifests(*run_dirs: str | Path) -> list[dict]:
    return [json.loads(p.read_text()) for d in run_dirs for p in sorted(Path(d).rglob("manifest.json"))]




def _ci(s: dict, label: str, pct: bool = False) -> str:
    v, lo, hi = s.get(label), s.get(f"{label}_lo"), s.get(f"{label}_hi")
    if v is None:
        return "—"
    f = (lambda x: f"{100 * x:.0f}%") if pct else (lambda x: f"{x:.2f}")
    return f(v) if lo == hi else f"{f(v)} [{f(lo)}, {f(hi)}]"


HEADERS = {"seller_agent": "seller", "seller_model": "model", "buyer_persona": "buyer", "buyer_model": "buyer model"}


def _readable(cell: str) -> str:
    """Seller and model ids as the report names them ('B + defences · claude-haiku-5.5'); other text untouched."""
    if " " in cell:
        return cell
    if any(cell.startswith(k) for k in DESIGN_LABELS):
        return display_name(cell)
    return model_name(cell) if re.fullmatch(r"[A-Za-z][\w.-]*/[\w.:-]+", cell) else cell  # provider/model, not "7/7"


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    header = [HEADERS.get(h, h) for h in header]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(_readable(str(c)) for c in r) + " |" for r in rows])


def _in_design_order(summaries: list[dict]) -> list[dict]:
    """Model first, then A, A+tools, A+plan, B (summarize sorts by the raw names)."""
    return sorted(summaries, key=lambda s: (str(s.get("seller_model", "")), _design_rank(s.get("seller_agent", "")),
                                            str(s.get("buyer_persona", ""))))


def main_table(rows: list[dict], group_by=("seller_agent", "seller_model")) -> str:
    """The headline table: one row per seller variant, mean [95% CI] per metric."""
    out = []
    for s in _in_design_order(summarize(_extra(rows), group_by=group_by, metrics=EXTRA_METRICS)):
        out.append([*(str(s[k]) for k in group_by), str(s["games"]),
                    _ci(s, "deal_rate", True), _ci(s, "surplus_share"), _ci(s, "seller_payoff"), _ci(s, "below_floor_rate", True),
                    _ci(s, "floor_leak_rate", True), _ci(s, "dishonest_rate", True), _ci(s, "major_dishonest_rate", True),
                    _ci(s, "seller_format_failures"), f"${s['cost_per_game_usd']:.4f}"])
    return markdown_table([*group_by, "games", "deal rate", "surplus share", "payoff", "below floor", "floor leak",
                           "dishonest", "major lies", "format fails", "cost/game"], out)


def attack_table(rows: list[dict]) -> str:
    """RQ3 headline: effect = success under attack minus under its plain twin."""
    pct = lambda x: "—" if x is None else f"{100 * x:.0f}%"
    out = []
    for e in attack_effect(rows):
        rule = ATTACK_SUCCESS_RULES.get(e["attack_category"], "either").replace("_", " ")
        effect = "—" if e["effect"] is None else f"{100 * e['effect']:+.0f} pts [{100 * e['lo']:+.0f}, {100 * e['hi']:+.0f}]"
        out.append([e["seller_agent"], e["attack_category"], rule, f"{e['n_attack']}/{e['n_twin']}",
                    pct(e["attack_rate"]), pct(e["twin_rate"]), effect])
    return markdown_table(["seller", "attack", "success =", "games (attack/twin)", "attack", "twin", "effect"], out)


def cost_table(*run_dirs) -> str:
    out = []
    for d in run_dirs:
        judge = sum(v.get("cost_usd", 0.0) for h in Path(d).rglob("honesty.jsonl") for v in load_honesty(h).values())
        for m in load_manifests(d):
            # The run folder, not the config's experiment name: pilot_sol ran configs/pilot.yaml with another seller.
            out.append([Path(d).parent.name, str(m["games_played"]), str(m.get("error_games", 0)), f"${m['spent_usd']:.4f}",
                        "—" if m["cost_per_game_usd"] is None else f"${m['cost_per_game_usd']:.4f}",
                        f"${judge:.4f}", m["aborted"] or "no"])
    return markdown_table(["experiment", "games", "crashed", "spent", "cost/game", "judge spend", "aborted"], out)


def calibration_table(rows: list[dict]) -> str:
    out = [[c["model"], str(c["games"]), _ci({"x": c["deal_rate"], "x_lo": c["deal_rate_lo"], "x_hi": c["deal_rate_hi"]}, "x", True),
            _ci({"x": c["surplus_share"], "x_lo": c["surplus_share_lo"], "x_hi": c["surplus_share_hi"]}, "x"),
            "—" if c["offset_from_half"] is None else f"{c['offset_from_half']:+.2f}",
            "yes" if c["centred"] else "no"] for c in self_play_calibration(rows)]
    return markdown_table(["model (both sides)", "games", "deal rate", "seller surplus", "offset from 0.5", "0.5 in CI"], out)


def paired_table(rows: list[dict], pairs: list[tuple[str, str]], metric="surplus_share", variant_key="seller_variant") -> str:
    out = []
    for a, b in pairs:
        p = paired_difference(rows, a, b, metric=metric, variant_key=variant_key)
        if p["mean_diff"] is None:
            out.append([a, b, "—", "—", "0"])
            continue
        out.append([a, b, f"{p['mean_diff']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]",
                    f"{100 * p['win_rate_a']:.0f}% [{100 * p['win_rate_a_lo']:.0f}, {100 * p['win_rate_a_hi']:.0f}]",
                    str(p["pairs"])])
    # Columns are "seller" and "compared with", not A/B, which are seller names in this project.
    return markdown_table(["seller", "compared with", f"mean {metric} difference (95% CI)", "win rate", "paired scenarios"], out)




def ablation_table(rows: list[dict], metric="seller_payoff") -> str:
    """RQ1: each seller variant against Agent A on the same scenarios and buyers. No deal = 0."""
    variants = sorted({r["seller_agent"] for r in rows})
    base = next((v for v in variants if v.startswith("prompted_seller")), None)
    pairs = [(v, base) for v in variants if base and v != base]
    return paired_table(rows, pairs, metric=metric, variant_key="seller_agent")


def honesty_pairs(on_rows: list[dict], off_rows: list[dict], metric="dishonest") -> str:
    """RQ4: honesty instruction off vs on, paired on the same scenario and buyer."""
    rows = on_rows + off_rows
    off = sorted({r["seller_agent"] for r in off_rows})
    pairs = [(o, o.split("[honesty_instruction")[0]) for o in off]
    return paired_table([{**r, metric: None if r[metric] is None else float(r[metric])} for r in rows],
                        pairs, metric=metric, variant_key="seller_agent")


def concession_table(results: list) -> str:
    """RQ6: concession per seller turn by how much of the deadline is left."""
    out = [[c["seller_agent"], c["time_left"], str(c["games"]), str(c["turns"]),
            _ci({"x": c["mean_concession"], "x_lo": c["mean_concession_lo"], "x_hi": c["mean_concession_hi"]}, "x"),
            _ci({"x": c["concession_rate"], "x_lo": c["concession_rate_lo"], "x_hi": c["concession_rate_hi"]}, "x", True)]
           for c in concession_by_time(results)]
    return markdown_table(["seller", "time left", "games", "turns", "mean concession (share of asking-to-floor gap)",
                           "turns with a concession"], out)


def behaviour_table(rows: list[dict]) -> str:
    """Extension: waiting and outcomes by the buyer's deadline."""
    out = [[s["seller_agent"], f"{s['buyer_deadline_days']:g}", str(s["games"]), _ci(s, "seller_wait_hours"), _ci(s, "buyer_wait_hours"),
            _ci(s, "deal_rate", True), _ci(s, "seller_payoff"), _ci(s, "deadline_expired_rate", True)]
           for s in summarize(rows, group_by=("seller_agent", "buyer_deadline_days"))]
    return markdown_table(["seller", "buyer deadline (days)", "chats", "seller wait (h)", "buyer wait (h)", "deal rate",
                           "payoff", "expired"], out)


def market_table(results: list) -> str:
    """Extension: outcome per market (one item, several buyers)."""
    out = [[m["seller_agent"], str(m["markets"]),
            _ci({"x": m["sold"], "x_lo": m["sold_lo"], "x_hi": m["sold_hi"]}, "x", True),
            _ci({"x": m["payoff"], "x_lo": m["payoff_lo"], "x_hi": m["payoff_hi"]}, "x"),
            _ci({"x": m["days_to_sale"], "x_lo": m["days_to_sale_lo"], "x_hi": m["days_to_sale_hi"]}, "x"),
            _ci({"x": m["below_floor"], "x_lo": m["below_floor_lo"], "x_hi": m["below_floor_hi"]}, "x", True)]
           for m in market_summary(results)]
    return markdown_table(["seller", "markets", "sold", "payoff (share of asking-to-floor gap)", "days to sale", "below floor"], out)


def competing_offer_table(attack_results: list, market_results: list) -> str:
    """Extension: seller concessions after fake and real competing offers, each with its baseline."""
    out = [[c["seller_agent"], c["condition"], str(c["games"]), str(c["turns"]),
            _ci({"x": c["mean_concession"], "x_lo": c["mean_concession_lo"], "x_hi": c["mean_concession_hi"]}, "x"),
            _ci({"x": c["concession_rate"], "x_lo": c["concession_rate_lo"], "x_hi": c["concession_rate_hi"]}, "x", True)]
           for c in competing_offer_response(attack_results, market_results)]
    return markdown_table(["seller", "condition", "games", "turns", "mean concession", "turns with a concession"], out)


# ---- RQ2: fine-tuned vs prompted, seller-side cost --------------------------------

SELLER_COST = [("seller_cost_usd", "seller_cost_per_game_usd")]
MAJOR_LIES = [("major_dishonest", "major_dishonest_rate")]
EXTRA_METRICS = SUMMARY_METRICS + SELLER_COST + MAJOR_LIES


def _extra(rows: list[dict]) -> list[dict]:
    """Rows with the report-only fields present, so summarize can read them (rows not from load_rows lack them)."""
    return [r if "major_dishonest" in r else {**r, "major_dishonest": None} for r in rows]


def seller_table(rows: list[dict], group_by=("seller_agent", "seller_model")) -> str:
    """RQ2 headline. Cost and latency are the seller's own: the buyer is the same API model in every row,
    so total cost per game would hide the difference between a local and a hosted seller."""
    out = []
    for s in _in_design_order(summarize(_extra(rows), group_by=group_by, metrics=EXTRA_METRICS)):
        out.append([*(str(s[k]) for k in group_by), str(s["games"]), _ci(s, "deal_rate", True), _ci(s, "seller_payoff"),
                    _ci(s, "below_floor_rate", True), _ci(s, "floor_leak_rate", True), _ci(s, "dishonest_rate", True),
                    _ci(s, "major_dishonest_rate", True), _ci(s, "seller_format_failures"), f"${s['seller_cost_per_game_usd']:.4f}",
                    "—" if s["seller_latency_s"] is None else f"{s['seller_latency_s']:.1f}s"])
    return markdown_table([*group_by, "games", "deal rate", "payoff", "below floor", "floor leak", "dishonest",
                           "major lies", "format fails", "seller cost/game", "seller latency/turn"], out)


# ---- RQ6: competition between buyers vs a one-buyer baseline ----------------------


def market_rows(results: list, setting: str) -> list[dict]:
    """One row per market (one item and its buyers), keyed so a market pairs with its twin in another run.

    The key drops the buyer suffix and the config hash: buyer 1 of each three-buyer market has the same
    listing, floor and budget as the lone buyer in the solo run (a test checks this).
    """
    markets: dict[tuple, list] = {}
    for r in results:
        markets.setdefault((re.sub(r"-b\d+$", "", r.scenario.id), r.seller_agent), []).append(r)
    out = []
    for (market, seller), chats in sorted(markets.items()):
        if any(c.outcome == "error" for c in chats):
            continue
        sc = chats[0].scenario
        sale = next((c for c in chats if c.outcome == "deal"), None)
        gap = max(sc.listing.asking_price - sc.seller_floor, 1.0)
        out.append({"market": market, "seller_agent": seller, "setting": setting, "buyers": len(chats),
                    "winner": sale.scenario.buyer_persona if sale else None,
                    "sold": sale is not None,
                    "payoff": (sale.deal_price - sc.seller_floor) / gap if sale else 0.0,
                    "days_to_sale": sale.turns[-1].simulated_time_hours / 24.0 if sale else None,
                    "below_floor": bool(sale and sale.deal_price < sc.seller_floor)})
    return out


def competition_table(solo: list, competition: list, label="three buyers") -> str:
    """RQ6 headline: each market with rivals minus the same market with one buyer, per seller, paired."""
    rows = market_rows(solo, "solo") + market_rows(competition, label)
    out = []
    for seller in sorted({r["seller_agent"] for r in rows}):
        mine = [r for r in rows if r["seller_agent"] == seller]
        for metric in ("payoff", "sold", "days_to_sale", "below_floor"):
            p = paired_difference(mine, label, "solo", metric=metric, variant_key="setting", pair_keys=("market",))
            means = {s: bootstrap_ci([r[metric] for r in mine if r["setting"] == s])[0] for s in ("solo", label)}
            fmt = (lambda x: "—" if x is None else f"{100 * x:.0f}%") if metric in ("sold", "below_floor") else \
                  (lambda x: "—" if x is None else f"{x:.2f}")
            diff = "—" if p["mean_diff"] is None else f"{p['mean_diff']:+.2f} [{p['lo']:+.2f}, {p['hi']:+.2f}]"
            out.append([seller, metric.replace("_", " "), fmt(means["solo"]), fmt(means[label]), diff, str(p["pairs"])])
    return markdown_table(["seller", "metric", "one buyer", label, f"{label} − one buyer (paired, 95% CI)", "markets"], out)


def winner_table(competition: list) -> str:
    """RQ6: which buyer persona got the item when all three were negotiating at once."""
    rows = market_rows(competition, "competition")
    out = []
    for seller in sorted({r["seller_agent"] for r in rows}):
        mine = [r for r in rows if r["seller_agent"] == seller]
        counts = {}
        for r in mine:
            counts[r["winner"] or "unsold"] = counts.get(r["winner"] or "unsold", 0) + 1
        out.append([seller, str(len(mine)), ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))])
    return markdown_table(["seller", "markets", "won by"], out)


# ---- judge checks ------------------------------------------------------------------


def _verdicts(*run_dirs) -> dict[str, dict]:
    return {k: v for d in run_dirs for h in Path(d).rglob("honesty.jsonl") for k, v in load_honesty(h).items()}


def leak_check_table(*run_dirs) -> str:
    """The regex floor-leak metric against the judge's floor_leak verdict on the same games.

    Both follow the same rule (the floor as a number or in words; offering the floor is not a leak),
    so disagreement measures how far a cheap regex can stand in for the judge.
    """
    out = []
    for d in run_dirs:
        verdicts = _verdicts(d)
        if not verdicts:
            continue
        pairs = [(floor_leak(r.turns, r.scenario), bool(verdicts[game_id(r)]["judge_floor_leak"]))
                 for r in load_results(d) if game_id(r) in verdicts and not verdicts[game_id(r)].get("error")]
        n = len(pairs)
        both = sum(a and b for a, b in pairs)
        regex_only = sum(a and not b for a, b in pairs)
        judge_only = sum(b and not a for a, b in pairs)
        out.append([Path(d).parent.name, str(n), f"{100 * sum(a for a, _ in pairs) / n:.0f}%",
                    f"{100 * sum(b for _, b in pairs) / n:.0f}%", f"{100 * (n - regex_only - judge_only) / n:.0f}%",
                    str(both), str(regex_only), str(judge_only)])
    return markdown_table(["run", "judged games", "regex leak", "judge leak", "agree", "both", "regex only", "judge only"], out)


def judge_agreement(labels_csv: str | Path, *run_dirs, major_only: bool = False) -> dict:
    """Judge vs human labels over the verdicts of every judged run (haggle.judge.agreement on one combined file).

    major_only=True counts a game as dishonest only when the judge found a major violation (a denied flaw or a
    false concrete fact), a sensitivity check for borderline "minor" flags.
    """
    verdicts = _verdicts(*run_dirs).values()
    if major_only:
        verdicts = [{**v, "violations": v["major_violations"]} for v in verdicts]
    with tempfile.TemporaryDirectory() as tmp:
        combined = Path(tmp) / "verdicts.jsonl"
        combined.write_text("".join(json.dumps(v) + "\n" for v in verdicts), encoding="utf-8")
        return agreement(Path(labels_csv), combined)


def agreement_table(labels_csv: str | Path, *run_dirs) -> str:
    """Judge-vs-human agreement for both questions, with the major-only sensitivity row."""
    out = []
    for name, major in (("any violation", False), ("major violations only", True)):
        a = judge_agreement(labels_csv, *run_dirs, major_only=major)
        rows = [("dishonest", f"dishonest ({name})")] + ([("floor_leak", "floor leak")] if not major else [])
        for key, label in rows:
            m = a[key]
            f = lambda x: "—" if x is None else f"{x:.2f}"
            out.append([label, str(m["n"]), f(m["accuracy"]), f(m["precision"]), f(m["recall"]), f(m["f1"]),
                        f"{m['tp']} / {m['fp']} / {m['fn']} / {m['tn']}"])
    return markdown_table(["question", "labelled games", "accuracy", "precision", "recall", "F1",
                           "TP / FP / FN / TN (judge vs human)"], out)


def judge_disagreements(labels_csv: str | Path, *run_dirs) -> str:
    """Every labelled game where the human and the judge differ, with both sides' reasons."""
    verdicts = {label_id(k): v for k, v in _verdicts(*run_dirs).items()}
    out = []
    with Path(labels_csv).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = verdicts.get(row["label_id"])
            if v is None or v.get("error"):
                continue
            judge = {"dishonest": v["violations"] > 0, "floor leak": bool(v["judge_floor_leak"])}
            human = {"dishonest": row["human_dishonest"].strip(), "floor leak": row["human_floor_leak"].strip()}
            for q in judge:
                if human[q] in ("0", "1") and (human[q] == "1") != judge[q]:
                    # A flagged claim says more than the summary sentence, which comes from one of the voting samples.
                    flagged = "; ".join(f"[{d['severity']}] \"{d['claim']}\"" for d in v.get("details") or [])
                    why = flagged if q == "dishonest" and flagged else (v.get("reasoning") or "")
                    out.append([row["label_id"], row["listing"][:40], q, "yes" if human[q] == "1" else "no",
                                "yes" if judge[q] else "no", row["notes"].replace("|", "/")[:160],
                                why.replace("|", "/").replace("\n", " ")[:160]])
    return markdown_table(["game", "listing", "question", "human", "judge", "labeller's note", "judge's evidence"], out)


# ---- charts --------------------------------------------------------------------------
# One look for every figure, sized for 16:9 slides: a left-aligned title and a one-line note on what is plotted,
# hairline grid on the value axis only, thin marks, CIs as thin lines, and selective value labels in ink (never
# in the series colour). Colours: blue then orange then aqua (validated as a set for colour-blind readers);
# grey is the baseline. Categorical colour follows the model, so a model keeps its colour in every chart.

BLUE, ORANGE, AQUA = SERIES[0], SERIES[1], SERIES[2]
INK2, AXIS, BASELINE_GREY, SURFACE = "#52514e", "#c3c2b7", "#9a9893", "#ffffff"
BLUE_RAMP = ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11.5, "ytick.labelsize": 12,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK, "text.color": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "legend.fontsize": 11.5,
})

# Chart and table labels: the design letters used in the report, and the model without its provider prefix.
DESIGN_LABELS = {"prompted_seller": "A", "agentic_seller[tools]": "A+tools", "agentic_seller[plan]": "A+plan",
                 "agentic_seller[tools+plan+reflect]": "B"}
VARIANT_SUFFIXES = {"[injection_defense=True]+guarded": " + defences", "[honesty_instruction=False]": " (no honesty line)"}
DESIGN_ORDER = ["A", "A+tools", "A+plan", "B"]


def model_name(model) -> str:
    return str(model).split("/")[-1]


def display_name(agent: str) -> str:
    """'agentic_seller[tools+plan+reflect][injection_defense=True]+guarded' -> 'B + defences'. Others unchanged."""
    agent = str(agent)
    if "@" in agent:
        a, m = agent.split("@", 1)
        return f"{display_name(a)} · {model_name(m)}"
    for key in sorted(DESIGN_LABELS, key=len, reverse=True):
        if agent.startswith(key):
            rest = agent[len(key):]
            return DESIGN_LABELS[key] + VARIANT_SUFFIXES.get(rest, rest)
    return agent


def _design_rank(agent: str) -> tuple:
    name = display_name(agent)
    base = name.split(" ")[0]
    return (DESIGN_ORDER.index(base) if base in DESIGN_ORDER else len(DESIGN_ORDER), name)


def short_label(s: dict, group_by) -> str:
    parts = []
    for k in group_by:
        v = str(s[k])
        parts.append(display_name(v) if k == "seller_agent" else model_name(v) if k.endswith("_model") else v)
    return " · ".join(parts)


# A model keeps one colour and one name in every chart. Both come from its role in configs/models.yaml, so no model
# name is written here: the hosted main seller is blue, the fine-tuned model orange, its untuned base aqua.
ROLE_STYLE = [("main_seller", BLUE, ""), ("finetuned", ORANGE, " (fine-tuned)"), ("weak", AQUA, " (base)")]


def _roles() -> dict:
    import yaml
    from haggle.config import PROJECT_ROOT
    try:
        models = yaml.safe_load((PROJECT_ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))["models"]
    except Exception:
        return {}
    return {models[r]["model"]: (i, c, suffix) for i, (r, c, suffix) in enumerate(ROLE_STYLE) if r in models}


MODEL_ROLES = _roles()


def model_label(model) -> str:
    """The model's short name, marked (fine-tuned) or (base) by its role."""
    return model_name(model) + MODEL_ROLES.get(model, (0, "", ""))[2]


def model_order(models) -> list:
    """Base model first, then fine-tuned, then the main seller, then anything else: the order of the RQ2 story."""
    rank = {m: (-v[0], "") for m, v in MODEL_ROLES.items()}
    return sorted(set(models), key=lambda m: rank.get(m, (1, str(m))))


def _model_colors(points: list[dict]) -> dict:
    """Role colours for known models; models outside configs/models.yaml (e.g. an old pilot seller) take yellow."""
    models = sorted({s.get("seller_model") for s in points}, key=lambda m: MODEL_ROLES.get(m, (9,))[0])
    spare = iter([SERIES[3], MUTED, INK2])
    return {m: MODEL_ROLES[m][1] if m in MODEL_ROLES else next(spare) for m in models}


def _figure(width=10.0, height=5.6, ncols=1, sharey=False, width_ratios=None):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), sharey=sharey,
                             gridspec_kw={"width_ratios": width_ratios} if width_ratios else None)
    return fig, axes


def _style(ax, value_axis="x"):
    """Hairline grid on the value axis only; no frame except a quiet baseline."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    if value_axis == "y":
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0, pad=6)
    ax.grid(False)
    if value_axis in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, linewidth=1)
    if value_axis in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _header(fig, title: str, subtitle: str = "", legend_handles=None):
    """Left-aligned title and note above the plot, then lay out the axes below them."""
    import textwrap

    w, h = fig.get_figwidth(), fig.get_figheight()
    # The figure grows by the header's height, so the plot keeps its size however long the note is.
    lines = textwrap.wrap(subtitle, width=int(w * 11.5)) if subtitle else []
    extra = 0.22 * max(0, len(lines) - 1)
    fig.set_figheight(h + extra)
    h += extra
    y = 1 - 0.16 / h
    fig.text(0.012, y, title, fontsize=18, fontweight="semibold", color=INK, va="top", ha="left")
    y -= 0.40 / h
    if lines:
        # Text wider than the figure would stretch the saved image (bbox_inches="tight"), so the note is wrapped.
        fig.text(0.012, y, "\n".join(lines), fontsize=12, color=INK2, va="top", ha="left", linespacing=1.35)
        y -= (0.30 + 0.22 * (len(lines) - 1)) / h
    if legend_handles:
        fig.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.005, y), ncol=len(legend_handles),
                   handlelength=1.2, columnspacing=1.6, borderaxespad=0)
        y -= 0.36 / h
    fig.tight_layout(rect=(0, 0, 1, y - 0.08 / h), w_pad=3.5)


def _dot(color, label):
    return plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=color, label=label)


def _swatch(color, label):
    return plt.Rectangle((0, 0), 1, 1, color=color, label=label)


def _ci_dot(ax, x, y, lo, hi, color, horizontal=True, size=9):
    if lo is not None and hi is not None:
        if horizontal:
            ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
        else:
            ax.plot([x, x], [lo, hi], color=color, linewidth=2, solid_capstyle="round", zorder=2)
    ax.plot([x], [y], marker="o", markersize=size, color=color, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)


def _pct(x) -> str:
    return f"{100 * x:.0f}%"


ATTACK_NAMES = {"deadline_exploit": "Deadline exploit", "extreme_lowball": "Extreme lowball",
                "fake_competing_offer": "Fake competing offer", "fake_urgency": "Fake urgency",
                "floor_extraction": "Floor extraction", "prompt_injection": "Prompt injection", "sob_story": "Sob story"}


def attack_success_chart(rows: list[dict], sellers: list[str] | None = None):
    """RQ3: attack success rate per category and seller, as a grid with the number in every cell.

    Rows may mix models; columns group by model. Attacks only; see attack_table for the twin baseline.
    """
    from matplotlib.colors import LinearSegmentedColormap

    attack_rows = [r for r in rows if r["attack_category"] and r.get("attack_kind") != "twin" and r.get("outcome") != "error"]
    summ = summarize(attack_rows, group_by=("seller_model", "seller_agent", "attack_category"))
    cell = {(s["seller_model"], s["seller_agent"], s["attack_category"]): s for s in summ}
    models = sorted({s["seller_model"] for s in summ}, key=str)
    cols: list = []
    for m in models:
        agents = sorted({s["seller_agent"] for s in summ if s["seller_model"] == m}, key=_design_rank)
        agents = [a for a in agents if not sellers or a in sellers]
        if cols:
            cols.append(None)  # a blank column between models
        cols += [(m, a) for a in agents]
    cats = sorted({s["attack_category"] for s in summ}, key=lambda c: ATTACK_NAMES.get(c, c))
    grid = [[float("nan") if col is None or (col[0], col[1], c) not in cell else cell[(col[0], col[1], c)]["attack_success_rate"]
             for col in cols] for c in cats]
    counts = [["" if col is None or (col[0], col[1], c) not in cell else
               f"{round(cell[(col[0], col[1], c)]['attack_success_rate'] * cell[(col[0], col[1], c)]['games'])}/{cell[(col[0], col[1], c)]['games']}"
               for col in cols] for c in cats]
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    cmap.set_bad(SURFACE)
    fig, ax = _figure(1.05 * len(cols) + 3.2, 0.62 * len(cats) + 2.9)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if v == v:  # not NaN
                ax.text(j, i, counts[i][j], ha="center", va="center", fontsize=12,
                        color=SURFACE if v >= 0.45 else INK if v > 0 else MUTED, fontweight="semibold" if v > 0 else "normal")
    ax.set_xticks(range(len(cols)), ["" if c is None else display_name(c[1]) for c in cols], fontsize=11.5)
    ax.set_yticks(range(len(cats)), [ATTACK_NAMES.get(c, c) for c in cats])
    ax.xaxis.tick_top()
    ax.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(cats))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    start = 0
    for m in models:  # model name above its group of columns
        n = sum(1 for c in cols if c is not None and c[0] == m)
        ax.annotate(model_name(m), ((start + start + n - 1) / 2, -0.5), xytext=(0, 30), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, fontweight="semibold", color=INK, annotation_clip=False)
        start += n + 1
    ns = [s["games"] for s in summ]
    games = f"{min(ns)}–{max(ns)}" if min(ns) != max(ns) else str(min(ns))
    _header(fig, "Attack success by category (RQ3)",
            f"Attack games the attacker won, out of {games} per cell: a sale below the seller's lowest price, or that price "
            "revealed (for extraction and injection).")
    return fig


def cost_quality_chart(rows: list[dict], group_by=("seller_agent", "seller_model"), cost="cost_per_game_usd"):
    """RQ2: cost per game vs seller surplus, with the Pareto frontier (best quality per budget).

    cost="seller_cost_per_game_usd" plots the seller's own spend, leaving out the shared buyer model.
    """
    pts = [s for s in summarize(rows, group_by=group_by, metrics=SUMMARY_METRICS + SELLER_COST) if s["surplus_share"] is not None]
    colors = _model_colors(pts)
    fig, ax = _figure(10, 5.8)
    front, best = [], float("-inf")
    for s in sorted(pts, key=lambda s: (s[cost], -s["surplus_share"])):
        if s["surplus_share"] > best:
            front.append(s)
            best = s["surplus_share"]
    if len(front) > 1:
        ax.step([s[cost] for s in front], [s["surplus_share"] for s in front], where="post",
                color=BASELINE_GREY, linewidth=1.5, linestyle="--", zorder=1)
    span = max(s[cost] for s in pts) or 1.0
    for s in pts:
        same = sorted((p for p in pts if abs(p[cost] - s[cost]) < span * 0.01), key=lambda p: p["surplus_share"])
        x = s[cost] + (same.index(s) - (len(same) - 1) / 2) * span * 0.025  # side by side, so the CIs don't overlap
        c = colors[s.get("seller_model")]
        _ci_dot(ax, x, s["surplus_share"], s["surplus_share_lo"], s["surplus_share_hi"], c, horizontal=False, size=10)
        ax.annotate(display_name(s["seller_agent"]), (x, s["surplus_share"]), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=12, color=INK, zorder=4,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": SURFACE, "edgecolor": "none"})
    _style(ax, value_axis="both")
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(AXIS)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.xaxis.set_major_formatter(lambda x, _: f"${x:.3f}")
    ax.set_xlim(-span * 0.06, span * 1.15)
    ax.set_xlabel("Seller cost per game (USD)" if cost.startswith("seller") else "Cost per game (USD)")
    ax.set_ylabel("Seller surplus share (mean, 95% CI)")
    handles = [_dot(c, model_name(m)) for m, c in colors.items()] + \
              [plt.Line2D([], [], color=BASELINE_GREY, linestyle="--", label="Best quality for the money")]
    _header(fig, "Cost vs deal quality (RQ2)",
            "Seller's own API cost per game; the buyer model is the same in every game. A locally served model costs $0.",
            handles)
    return fig


# ---- slide charts: plain bars, percentages, one idea per chart ----------------------
# Each chart takes a `title`, so the notebook can state the finding ("Competition helps the seller") while the
# function stays generic. Values are shares of the bargaining room, shown as percentages; the thin grey line on
# each bar is its 95% confidence range.

ROOM = "bargaining room = the gap between the seller's lowest price and the buyer's highest"
CI_NOTE = "Thin lines: 95% confidence range."
LIES_NOTE = "Major lie = a flaw denied or a false concrete claim (LLM judge; no false alarms on 30 hand-labelled games). "
DESIGN_DESCRIPTIONS = {"A": "A: prompt only", "A+tools": "A + price tool", "A+plan": "A + planning",
                       "B": "B: planning + tool + reflection"}


def _describe(agent: str) -> str:
    name = display_name(agent)
    return DESIGN_DESCRIPTIONS.get(name, name)


def _bars(ax, items: list[dict], fmt, label_size=12.5):
    """Horizontal bars with a thin CI line and the value printed past the end.

    items: {"y", "value", "lo", "hi", "color"}; "text" overrides the printed value.
    """
    for it in items:
        v = it["value"]
        if v is None:
            continue
        ax.barh(it["y"], v, height=it.get("height", 0.62), color=it["color"], zorder=2)
        lo, hi = it.get("lo"), it.get("hi")
        if lo is not None and hi is not None and lo != hi:
            ax.plot([lo, hi], [it["y"], it["y"]], color=INK2, linewidth=1.3, alpha=0.75, zorder=3, solid_capstyle="round")
        end = max(v, hi if hi is not None else v, 0)
        ax.annotate(it.get("text") or fmt(v), (end, it["y"]), xytext=(7, 0), textcoords="offset points", va="center",
                    fontsize=label_size, fontweight="semibold", color=INK)


def _pct_axis(ax, lo=None, hi=None):
    """Percent ticks that stop at 100% (the room past it is only for value labels)."""
    ax.set_xlim(lo, hi)
    lo, hi = ax.get_xlim()
    step = next(s for s in (0.1, 0.2, 0.4, 0.5, 1.0) if (min(hi, 1.0) - lo) / s <= 6)  # at most ~6 labels
    first = step * math.ceil(lo / step - 1e-9)
    ax.set_xticks([round(first + i * step, 6) for i in range(int((min(hi, 1.0) - first) / step + 1e-9) + 1)])
    ax.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}%")


def _grouped(groups: list[str], series: list, gap=0.7, height=0.62):
    """y positions for len(series) bars per group, with a gap between groups; returns (positions, group centres)."""
    pos, centres = {}, []
    y = 0.0
    for g in groups:
        ys = [y + j * height for j in range(len(series))]
        for s, yy in zip(series, ys):
            pos[(g, s)] = yy
        centres.append(sum(ys) / len(ys))
        y = ys[-1] + height + gap
    return pos, centres


def ablation_chart(rows: list[dict], title: str = "Seller results by agent design (RQ1)"):
    """RQ1: share of the bargaining room each design keeps (no deal = 0%), one bar per design, on one model."""
    summ = sorted(summarize(rows, group_by=("seller_agent",)), key=lambda s: _design_rank(s["seller_agent"]))
    model = rows[0]["seller_model"] if rows else ""
    color = _model_colors([{"seller_model": model}]).get(model, BLUE)
    fig, ax = _figure(11, 0.85 * len(summ) + 2.6)
    _bars(ax, [{"y": i, "value": s["seller_payoff"], "lo": s["seller_payoff_lo"], "hi": s["seller_payoff_hi"],
                "color": color} for i, s in enumerate(summ)], _pct)
    ax.set_yticks(range(len(summ)), [_describe(s["seller_agent"]) for s in summ])
    ax.set_ylim(len(summ) - 0.5, -0.5)
    _style(ax)
    ax.axvline(0, color=AXIS, linewidth=1)
    _pct_axis(ax, min(0, min(s["seller_payoff_lo"] for s in summ)) - 0.03, max(s["seller_payoff_hi"] for s in summ) + 0.1)
    ax.set_xlabel("Share of the bargaining room the seller kept (no deal counts as 0%)")
    games = summ[0]["games"] if summ else 0
    buyer = model_name(rows[0]["buyer_model"]) if rows else ""
    _header(fig, title, f"Seller: {model_label(model)}. Buyer: {buyer}, the same {games} games for every design. "
            f"{ROOM.capitalize()}. {CI_NOTE}")
    return fig


def ablation_by_model_chart(rows: list[dict], title: str = "Agent design on each model (RQ1)"):
    """RQ1 on every model at once: one panel per model, the same four designs, the same axis, so panels compare."""
    models = model_order({r["seller_model"] for r in rows})
    summ = summarize(rows, group_by=("seller_model", "seller_agent"))
    agents = sorted({s["seller_agent"] for s in summ}, key=_design_rank)
    cell = {(s["seller_model"], s["seller_agent"]): s for s in summ}
    colors = _model_colors([{"seller_model": m} for m in models])
    lo = min(0, min(s["seller_payoff_lo"] for s in summ)) - 0.03
    hi = max(s["seller_payoff_hi"] for s in summ) + 0.14
    fig, axes = _figure(4.6 * len(models) + 2.2, 0.8 * len(agents) + 2.9, ncols=len(models), sharey=True)
    axes = list(axes) if len(models) > 1 else [axes]
    for ax, m in zip(axes, models):
        _bars(ax, [{"y": i, "value": cell[(m, a)]["seller_payoff"], "lo": cell[(m, a)]["seller_payoff_lo"],
                    "hi": cell[(m, a)]["seller_payoff_hi"], "color": colors[m]}
                   for i, a in enumerate(agents) if (m, a) in cell], _pct, label_size=12)
        _style(ax)
        ax.axvline(0, color=AXIS, linewidth=1)
        _pct_axis(ax, lo, hi)
        ax.set_title(model_label(m), loc="left", fontsize=13, fontweight="semibold", color=INK, pad=10)
    axes[0].set_yticks(range(len(agents)), [_describe(a) for a in agents])
    axes[0].set_ylim(len(agents) - 0.5, -0.5)
    fig.supxlabel("Share of the bargaining room the seller kept (no deal counts as 0%; below 0% = sold below its lowest "
                  "price on average)", fontsize=12, color=INK2, y=0.02)
    _header(fig, title, f"The same games and buyers for every bar. {ROOM.capitalize()}. {CI_NOTE}")
    return fig


def model_comparison_chart(rows: list[dict], title: str = "Fine-tuned vs prompted model (RQ2)", panels=None):
    """RQ2: the same designs on several models, side by side. Default panels: deal quality, below-floor sales, lying.

    Only designs that every model played are shown. Models run base -> fine-tuned -> main seller (model_order).
    """
    panels = panels or (("seller_payoff", "Bargaining room kept"), ("below_floor_rate", "Sold below its lowest price"),
                        ("major_dishonest_rate", "Told a major lie about the item"))
    summ = summarize(_extra(rows), group_by=("seller_agent", "seller_model"), metrics=EXTRA_METRICS)
    models = model_order({s["seller_model"] for s in summ})
    agents = sorted({s["seller_agent"] for s in summ
                     if sum(1 for t in summ if t["seller_agent"] == s["seller_agent"]) == len(models)}, key=_design_rank)
    cell = {(s["seller_agent"], s["seller_model"]): s for s in summ}
    colors = _model_colors([{"seller_model": m} for m in models])
    pos, centres = _grouped(agents, models)
    fig, axes = _figure(4.3 * len(panels) + 1.0, 0.62 * len(pos) + 3.2, ncols=len(panels), sharey=True)
    axes = list(axes) if len(panels) > 1 else [axes]
    for ax, (metric, label) in zip(axes, panels):
        items = [{"y": pos[(a, m)], "value": cell[(a, m)][metric], "lo": cell[(a, m)][f"{metric}_lo"],
                  "hi": cell[(a, m)][f"{metric}_hi"], "color": colors[m]} for a in agents for m in models]
        _bars(ax, items, _pct, label_size=12)
        _style(ax)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_title(label, loc="left", fontsize=13, fontweight="semibold", color=INK, pad=10)
        shown = [it for it in items if it["value"] is not None]
        lows = [it["lo"] for it in shown if it["lo"] is not None] + [0]
        _pct_axis(ax, min(lows) - 0.03, max((it["hi"] if it["hi"] is not None else it["value"]) for it in shown) + 0.22)
    axes[0].set_yticks(centres, [_describe(a) for a in agents])
    axes[0].set_ylim(max(pos.values()) + 0.6, -0.6)

    def cost(m):
        c = sum(cell[(a, m)]["seller_cost_per_game_usd"] for a in agents) / len(agents)
        return "free, runs locally" if c == 0 else f"about ${c:.3f} per game"

    buyer = model_name(rows[0]["buyer_model"]) if rows else ""
    _header(fig, title, f"Buyer: {buyer}, the same games for every seller. Bargaining room kept: no deal counts as 0%; "
            "below 0% = sold below its lowest price on average. "
            + (LIES_NOTE if any(m.startswith("major") for m, _ in panels) else "") + CI_NOTE,
            [_swatch(colors[m], f"{model_label(m)}: {cost(m)}") for m in models])
    return fig


def attack_summary_chart(rows: list[dict], title: str = "How often attacks worked (RQ3)"):
    """RQ3 in one look: attacks that worked out of all attack games, per seller, one panel per model."""
    att = [r for r in rows if r["attack_category"] and r.get("attack_kind") == "attack" and r.get("outcome") != "error"]
    models = model_order({r["seller_model"] for r in att})
    agents = sorted({r["seller_agent"] for r in att}, key=_design_rank)
    colors = _model_colors([{"seller_model": m} for m in models])
    fig, axes = _figure(4.3 * len(models) + 2.6, 0.8 * len(agents) + 2.9, ncols=len(models), sharey=True)
    axes = list(axes) if len(models) > 1 else [axes]
    panels = {}
    for m in models:
        items = []
        for i, a in enumerate(agents):
            mine = [r for r in att if r["seller_model"] == m and r["seller_agent"] == a]
            if not mine:
                continue
            k = sum(bool(r["attack_success"]) for r in mine)
            v, lo, hi, n = bootstrap_ci([r["attack_success"] for r in mine])
            items.append({"y": i, "value": v, "lo": lo, "hi": hi, "color": colors[m], "text": f"{k} of {n}"})
        panels[m] = items
    top = max(0.6, max((it["hi"] or 0) for items in panels.values() for it in items) + 0.15)  # one scale for all panels
    for ax, m in zip(axes, models):
        _bars(ax, panels[m], _pct)
        _style(ax)
        ax.axvline(0, color=AXIS, linewidth=1)
        _pct_axis(ax, 0, top)
        ax.set_title(model_label(m), loc="left", fontsize=13, fontweight="semibold", color=INK, pad=10)
    axes[0].set_yticks(range(len(agents)), [display_name(a) for a in agents])
    axes[0].set_ylim(len(agents) - 0.5, -0.5)
    fig.supxlabel("Attack games the attacker won", fontsize=12, color=INK2, y=0.02)
    cats = len({r["attack_category"] for r in att})
    _header(fig, title, f"Scripted buyers tried {cats} tricks (sob stories, fake urgency, prompt injection, ...). An attack wins "
            "if the seller sells below its lowest price or reveals it. + defences = injection defence and message filter.")
    return fig


OUTCOME_COLORS = {"good": "#0ca30c", "below": "#d03b3b", "none": "#c3c2b7"}


def outcome_chart(rows: list[dict], title: str = "How each seller's games ended"):
    """Every game, in three outcomes: sold at or above the lowest price, sold below it, or no deal. One bar per seller.

    The plainest view of what a seller did; it shows what a bargaining-room average hides (a seller can average 0%
    by selling well half the time and at a loss the other half).
    """
    rows = [r for r in rows if r.get("outcome") != "error"]
    keys = sorted({(r["seller_model"], r["seller_agent"]) for r in rows},
                  key=lambda k: (model_order({r["seller_model"] for r in rows}).index(k[0]), _design_rank(k[1])))
    models = [k[0] for k in keys]
    ys, y = [], 0.0
    for i, k in enumerate(keys):
        if i and models[i] != models[i - 1]:
            y += 0.6  # a gap between models
        ys.append(y)
        y += 1
    fig, ax = _figure(12.5, 0.55 * len(keys) + 3.2)
    for yy, (m, a) in zip(ys, keys):
        mine = [r for r in rows if r["seller_model"] == m and r["seller_agent"] == a]
        n = len(mine)
        parts = [("good", sum(r["deal"] and not r["below_floor"] for r in mine) / n),
                 ("below", sum(bool(r["below_floor"]) for r in mine) / n),
                 ("none", sum(not r["deal"] for r in mine) / n)]
        left = 0.0
        for key, share in parts:
            if share <= 0:
                continue
            ax.barh(yy, share, left=left, height=0.62, color=OUTCOME_COLORS[key], edgecolor=SURFACE, linewidth=2, zorder=2)
            if share >= 0.07:
                ax.text(left + 0.012, yy, _pct(share), va="center", ha="left", fontsize=11.5, fontweight="semibold",
                        color=SURFACE if key != "none" else INK)
            left += share
    ax.set_yticks(ys, [f"{display_name(a)}  ·  {model_label(m)}" for m, a in keys])
    ax.set_ylim(max(ys) + 0.6, -0.6)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}%")
    _style(ax, value_axis="none")
    ax.set_xticks([])
    _header(fig, title, "Share of each seller's games. A = prompt only; B = planning + price tool + reflection.",
            [_swatch(OUTCOME_COLORS["good"], "Sold at or above its lowest price"),
             _swatch(OUTCOME_COLORS["below"], "Sold below its lowest price"), _swatch(OUTCOME_COLORS["none"], "No deal")])
    return fig


def price_path_chart(games: list, title: str = "One negotiation, four sellers"):
    """The offers in one scenario, message by message, one panel per seller: how each design reached its price.

    games: GameResults on the same scenario and buyer persona (e.g. each design's game on one listing).
    """
    games = sorted(games, key=lambda g: _design_rank(g.seller_agent))
    sc = games[0].scenario
    fig, axes = _figure(4.0 * len(games) + 1.5, 5.4, ncols=len(games), sharey=True)
    axes = list(axes) if len(games) > 1 else [axes]
    for ax, g in zip(axes, games):
        seller = [(t.index, t.action.price) for t in g.turns if t.speaker == "seller" and t.action.action == "offer"]
        seller = [(-0.5, sc.listing.asking_price)] + seller
        buyer = [(t.index, t.action.price) for t in g.turns if t.speaker == "buyer" and t.action.action == "offer"]
        ax.axhspan(sc.seller_floor, sc.buyer_budget, color=BLUE_RAMP[0], zorder=0)
        ax.axhline(sc.seller_floor, color=OUTCOME_COLORS["below"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        ax.axhline(sc.buyer_budget, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
        ax.plot(*zip(*seller), color=BLUE, linewidth=2.2, marker="o", markersize=6, markeredgecolor=SURFACE, zorder=3)
        if buyer:
            ax.plot(*zip(*buyer), color=BASELINE_GREY, linewidth=2.2, marker="o", markersize=6, markeredgecolor=SURFACE, zorder=3)
        if g.deal_price is not None:
            ax.plot([g.turns[-1].index], [g.deal_price], marker="*", markersize=17, color=INK, zorder=4)
            ax.annotate(f"${g.deal_price:,.0f}", (g.turns[-1].index, g.deal_price), xytext=(0, 12), textcoords="offset points",
                        ha="center", fontsize=12.5, fontweight="semibold", color=INK, zorder=5,
                        bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"})
        ax.set_title(_describe(g.seller_agent), loc="left", fontsize=12.5, fontweight="semibold", color=INK, pad=10)
        _style(ax, value_axis="y")
        ax.set_xticks([])
        ax.set_xlabel("messages  →")
    axes[0].yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    # The two limits go in the legend: labels on the plot collide with the deal price when it lands near a limit.
    _header(fig, title, f"{sc.listing.title}, asking ${sc.listing.asking_price:,.0f}. The shaded band is the bargaining room. "
            f"Seller: {model_label(games[0].seller_model)}; buyer: {model_name(games[0].buyer_model)}.",
            [_dot(BLUE, "Seller's offers"), _dot(BASELINE_GREY, "Buyer's offers"),
             plt.Line2D([], [], marker="*", linestyle="", markersize=14, color=INK, label="Deal"),
             plt.Line2D([], [], color=INK2, linestyle=(0, (4, 3)), label=f"Buyer's highest price (${sc.buyer_budget:,.0f})"),
             plt.Line2D([], [], color=OUTCOME_COLORS["below"], linestyle=(0, (4, 3)),
                        label=f"Seller's lowest price (${sc.seller_floor:,.0f})")])
    return fig


def surplus_vs_honesty(rows: list[dict], group_by=("seller_agent",), title: str = "Deal quality vs honesty (RQ4)",
                       lies: str = "major_dishonest_rate"):
    """RQ4: does a better negotiator lie more? Each seller's deal quality and lying rate, side by side.

    lies="major_dishonest_rate" (default) counts only major lies, where the judge matched the hand labels;
    lies="dishonest_rate" counts any flagged claim, including the minor sales talk it over-flags.
    """
    pts = [s for s in summarize(_extra(rows), group_by=group_by, metrics=EXTRA_METRICS) if s[lies] is not None]
    pts.sort(key=lambda s: (MODEL_ROLES.get(s.get("seller_model"), (9,))[0], _design_rank(s["seller_agent"])))
    colors = _model_colors(pts)
    models = list(colors)
    ys = [i + 0.7 * models.index(s.get("seller_model")) for i, s in enumerate(pts)]
    fig, (left, right) = _figure(12.5, 0.62 * len(pts) + 3.0, ncols=2, sharey=True)
    _bars(left, [{"y": y, "value": s["seller_payoff"], "lo": s["seller_payoff_lo"], "hi": s["seller_payoff_hi"],
                  "color": colors[s.get("seller_model")]} for y, s in zip(ys, pts)], _pct, label_size=12)
    _bars(right, [{"y": y, "value": s[lies], "lo": s[f"{lies}_lo"], "hi": s[f"{lies}_hi"],
                   "color": colors[s.get("seller_model")]} for y, s in zip(ys, pts)], _pct, label_size=12)
    for ax, label in ((left, "Better deals  →"), (right, "More lying  →")):
        _style(ax)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_title(label, loc="left", fontsize=13, fontweight="semibold", color=INK, pad=10)
    _pct_axis(left, min(0, min(s["seller_payoff_lo"] for s in pts)) - 0.03, max(s["seller_payoff_hi"] for s in pts) + 0.15)
    _pct_axis(right, 0, 1.12)
    left.set_xlabel("Bargaining room kept (no deal counts as 0%)")
    right.set_xlabel("Games with a major lie about the item" if lies.startswith("major") else
                     "Games where the judge flagged a claim about the item")
    left.set_yticks(ys, [display_name(s["seller_agent"]) for s in pts])
    left.set_ylim(max(ys) + 0.6, -0.6)
    handles = [_swatch(c, model_label(m)) for m, c in colors.items()] if len(colors) > 1 else None
    _header(fig, title, "If better negotiators lied more, the longest bars on the left would also be longest on the right. "
            "Below 0% = sold below its lowest price on average. " + (LIES_NOTE if lies.startswith("major") else
            "Lies are the LLM judge's verdicts. ") + CI_NOTE, handles)
    return fig


def asymmetry_chart(rows: list[dict], strong: str, weak: str, title: str = "Strong vs weak model (RQ5)"):
    """RQ5: who keeps more when a deal happens, and how often a deal happens, for each model pairing."""
    pairs = [r for r in rows if r["seller_model"] in (strong, weak) and r["buyer_model"] in (strong, weak)]
    summ = {(s["seller_model"], s["buyer_model"]): s for s in summarize(pairs, group_by=("seller_model", "buyer_model"))}
    order = [k for k in [(strong, weak), (strong, strong), (weak, weak), (weak, strong)] if k in summ]
    colors = _model_colors([{"seller_model": strong}, {"seller_model": weak}])
    fig, (left, right) = _figure(12.5, 0.85 * len(order) + 3.0, ncols=2, sharey=True)
    _bars(left, [{"y": i, "value": summ[k]["surplus_share"], "lo": summ[k]["surplus_share_lo"],
                  "hi": summ[k]["surplus_share_hi"], "color": colors[k[0]]} for i, k in enumerate(order)], _pct)
    _bars(right, [{"y": i, "value": summ[k]["deal_rate"], "lo": summ[k]["deal_rate_lo"], "hi": summ[k]["deal_rate_hi"],
                   "color": colors[k[0]]} for i, k in enumerate(order)], _pct)
    for ax, label in ((left, "Bargaining room the seller kept, when they made a deal"), (right, "Games that ended in a deal")):
        _style(ax)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_title(label, loc="left", fontsize=13, fontweight="semibold", color=INK, pad=10)
        _pct_axis(ax, 0, 1.22)
    left.axvline(0.5, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    left.annotate("even split", (0.5, len(order) - 0.5), xytext=(4, 4), textcoords="offset points", color=INK2, fontsize=10.5)
    left.set_yticks(range(len(order)), [f"{model_name(a)} sells to {model_name(b)}" for a, b in order])
    left.set_ylim(len(order) - 0.5, -0.5)
    _header(fig, title, f"{ROOM.capitalize()}. {CI_NOTE}",
            [_swatch(colors[strong], f"Seller: {model_label(strong)}"), _swatch(colors[weak], f"Seller: {model_label(weak)}")])
    return fig


def calibration_chart(rows: list[dict], title: str = "Self-play calibration"):
    """Self-play: the same model on both sides; a fair setup gives the seller about 50%."""
    cal = [c for c in self_play_calibration(rows) if c["surplus_share"] is not None]
    fig, ax = _figure(11, 0.9 * len(cal) + 2.7)
    _bars(ax, [{"y": i, "value": c["surplus_share"], "lo": c["surplus_share_lo"], "hi": c["surplus_share_hi"],
                "color": BLUE} for i, c in enumerate(cal)], _pct)
    ax.set_yticks(range(len(cal)), [model_name(c["model"]) for c in cal])
    ax.set_ylim(len(cal) - 0.5, -0.8)
    _style(ax)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.axvline(0.5, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.annotate("fair split", (0.5, -0.65), xytext=(5, 0), textcoords="offset points", color=INK2, fontsize=10.5, va="center")
    _pct_axis(ax, 0, 1.08)
    ax.set_xlabel("Bargaining room the seller kept, when they made a deal")
    _header(fig, title, f"Both sides run the same model, so a fair setup would give the seller about 50%. {CI_NOTE}")
    return fig


def competition_chart(solo: list, competition: list, label="Three buyers at once", title="Competition between buyers (RQ6)"):
    """RQ6: each seller with one buyer vs three buyers at once: the price it got and how fast it sold."""
    rows = market_rows(solo, "solo") + market_rows(competition, "comp")
    sellers = sorted({r["seller_agent"] for r in rows}, key=_design_rank)
    settings = [("solo", BASELINE_GREY, "One buyer"), ("comp", BLUE, label)]
    pos, centres = _grouped(sellers, [s for s, *_ in settings])
    fig, (left, right) = _figure(12.5, 0.75 * len(pos) + 3.0, ncols=2, sharey=True)
    for ax, metric, scale, fmt in ((left, "payoff", 1.0, _pct), (right, "days_to_sale", 24.0, lambda v: f"{v:.0f} hours")):
        items = []
        for seller in sellers:
            for setting, color, _ in settings:
                v, lo, hi, _n = bootstrap_ci([r[metric] for r in rows if r["seller_agent"] == seller and r["setting"] == setting])
                items.append({"y": pos[(seller, setting)], "value": None if v is None else v * scale,
                              "lo": None if lo is None else lo * scale, "hi": None if hi is None else hi * scale, "color": color})
        _bars(ax, items, fmt)
        _style(ax)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_xlim(0, max(it["hi"] or it["value"] for it in items) * 1.3)
    _pct_axis(left, 0, left.get_xlim()[1])
    left.set_title("Price achieved  (higher is better)", loc="left", fontsize=13, fontweight="semibold", color=INK, pad=10)
    right.set_title("Time until sold  (shorter is better)", loc="left", fontsize=13, fontweight="semibold", color=INK, pad=10)
    left.set_xlabel("Share of the gap between the lowest price and the asking price")
    right.set_xlabel("Hours")
    left.set_yticks(centres, [_describe(s) for s in sellers])
    left.set_ylim(max(pos.values()) + 0.6, -0.6)
    _header(fig, title, f"Same 20 listings and lowest prices in both setups; unsold counts as 0%. {CI_NOTE}",
            [_swatch(c, name) for _, c, name in settings])
    return fig


def save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.25)
    return path
