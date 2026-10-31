"""Report tables and charts, built from run directories. Used by analysis/results.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from haggle.harnesses import paired_difference, self_play_calibration
from haggle.metrics import (
    ATTACK_SUCCESS_RULES,
    attack_effect,
    game_id,
    game_metrics,
    load_honesty,
    load_results,
    summarize,
    surplus_shift,
)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"




def load_rows(*run_dirs: str | Path) -> list[dict]:
    """Metric rows for every game under the given run dirs, joined with judge verdicts."""
    rows = []
    for d in run_dirs:
        d = Path(d)
        verdicts = {}
        for h in d.rglob("honesty.jsonl"):
            verdicts |= load_honesty(h)
        for r in load_results(d):
            v = verdicts.get(game_id(r))
            rows.append(game_metrics(r, v if v and v.get("violations") is not None else None))
    return rows


def load_manifests(*run_dirs: str | Path) -> list[dict]:
    return [json.loads(p.read_text()) for d in run_dirs for p in sorted(Path(d).rglob("manifest.json"))]




def _ci(s: dict, label: str, pct: bool = False) -> str:
    v, lo, hi = s.get(label), s.get(f"{label}_lo"), s.get(f"{label}_hi")
    if v is None:
        return "—"
    f = (lambda x: f"{100 * x:.0f}%") if pct else (lambda x: f"{x:.2f}")
    return f(v) if lo == hi else f"{f(v)} [{f(lo)}, {f(hi)}]"


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def main_table(rows: list[dict], group_by=("seller_agent", "seller_model")) -> str:
    """The headline table: one row per seller variant, mean [95% CI] per metric."""
    out = []
    for s in summarize(rows, group_by=group_by):
        out.append([*(str(s[k]) for k in group_by), str(s["games"]),
                    _ci(s, "deal_rate", True), _ci(s, "surplus_share"), _ci(s, "seller_payoff"), _ci(s, "below_floor_rate", True),
                    _ci(s, "floor_leak_rate", True), _ci(s, "dishonest_rate", True),
                    _ci(s, "seller_format_failures"), f"${s['cost_per_game_usd']:.4f}"])
    return markdown_table([*group_by, "games", "deal rate", "surplus share", "payoff", "below floor", "floor leak",
                           "dishonest", "format fails", "cost/game"], out)


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
    out = [[m["experiment"], str(m["games_played"]), f"${m['spent_usd']:.4f}",
            "—" if m["cost_per_game_usd"] is None else f"${m['cost_per_game_usd']:.4f}", m["aborted"] or "no"]
           for m in load_manifests(*run_dirs)]
    return markdown_table(["experiment", "games", "spent", "cost/game", "aborted"], out)


def calibration_table(rows: list[dict]) -> str:
    out = [[c["model"], str(c["games"]), _ci({"x": c["deal_rate"], "x_lo": c["deal_rate_lo"], "x_hi": c["deal_rate_hi"]}, "x", True),
            _ci({"x": c["surplus_share"], "x_lo": c["surplus_share_lo"], "x_hi": c["surplus_share_hi"]}, "x"),
            "—" if c["offset_from_half"] is None else f"{c['offset_from_half']:+.2f}",
            "yes" if c["centred"] else "no"] for c in self_play_calibration(rows)]
    return markdown_table(["model (both sides)", "games", "deal rate", "seller surplus", "offset from 0.5", "0.5 in CI"], out)


def paired_table(rows: list[dict], pairs: list[tuple[str, str]], metric="surplus_share") -> str:
    out = []
    for a, b in pairs:
        p = paired_difference(rows, a, b, metric=metric)
        if p["mean_diff"] is None:
            out.append([a, b, "—", "—", "0"])
            continue
        out.append([a, b, f"{p['mean_diff']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}]",
                    f"{100 * p['win_rate_a']:.0f}% [{100 * p['win_rate_a_lo']:.0f}, {100 * p['win_rate_a_hi']:.0f}]",
                    str(p["pairs"])])
    return markdown_table(["A", "B", f"mean {metric} A − B", "A win rate", "paired scenarios"], out)




def _axes(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _err(s: dict, label: str):
    v = s[label]
    return [[v - s[f"{label}_lo"]], [s[f"{label}_hi"] - v]]


def surplus_vs_honesty(rows: list[dict], group_by=("seller_agent",)):
    """RQ4: does a better negotiator lie more? One point per seller variant, CIs on both axes."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    _axes(ax, "Deal quality vs honesty (RQ4)", "Seller surplus share (mean, 95% CI)", "Games with an honesty violation")
    for i, s in enumerate(summarize(rows, group_by=group_by)):
        if s["surplus_share"] is None or s["dishonest_rate"] is None:
            continue
        name = " / ".join(str(s[k]) for k in group_by)
        ax.errorbar(s["surplus_share"], s["dishonest_rate"], xerr=_err(s, "surplus_share"), yerr=_err(s, "dishonest_rate"),
                    fmt="o", ms=8, color=SERIES[i % 3], ecolor=SERIES[i % 3], elinewidth=1.5, capsize=0, label=name)
        ax.annotate(name, (s["surplus_share"], s["dishonest_rate"]), xytext=(8, 6), textcoords="offset points", color=INK, fontsize=9)
    ax.yaxis.set_major_formatter(lambda y, _: f"{100 * y:.0f}%")
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    return fig


def attack_success_chart(rows: list[dict], sellers: list[str] | None = None):
    """RQ3: attack success rate by category, one bar per seller (max three sellers). Attacks only; see attack_table for the twin baseline."""
    attack_rows = [r for r in rows if r["attack_category"] and r.get("attack_kind") != "twin"]
    summ = summarize(attack_rows, group_by=("seller_agent", "attack_category"))
    sellers = (sellers or sorted({s["seller_agent"] for s in summ}))[:3]
    cats = sorted({s["attack_category"] for s in summ})
    fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(cats) * len(sellers) + 1.5))
    _axes(ax, "Attack success rate by category (RQ3)", "Success rate (95% CI)")
    h = 0.8 / max(1, len(sellers))
    for j, seller in enumerate(sellers):
        by_cat = {s["attack_category"]: s for s in summ if s["seller_agent"] == seller}
        ys = [i + j * h for i, c in enumerate(cats) if c in by_cat]
        ss = [by_cat[c] for c in cats if c in by_cat]
        ax.barh(ys, [s["attack_success_rate"] for s in ss], height=h, color=SERIES[j], label=seller,
                edgecolor="white", linewidth=2,
                xerr=[[s["attack_success_rate"] - s["attack_success_rate_lo"] for s in ss],
                      [s["attack_success_rate_hi"] - s["attack_success_rate"] for s in ss]],
                error_kw={"ecolor": INK, "elinewidth": 1})
    ax.set_yticks([i + h * (len(sellers) - 1) / 2 for i in range(len(cats))], [c.replace("_", " ") for c in cats])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}%")
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


def asymmetry_chart(rows: list[dict], strong: str, weak: str):
    """RQ5: seller surplus share for each (seller model → buyer model) pairing."""
    deals = [r for r in rows if r["seller_model"] in (strong, weak) and r["buyer_model"] in (strong, weak)]
    summ = {(s["seller_model"], s["buyer_model"]): s for s in summarize(deals, group_by=("seller_model", "buyer_model"))}
    order = [(strong, weak), (strong, strong), (weak, weak), (weak, strong)]
    order = [k for k in order if k in summ and summ[k]["surplus_share"] is not None]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    shift = surplus_shift(rows, strong, weak)
    sub = "" if shift["shift"] is None else f"  (shift {shift['shift']:+.2f} [{shift['lo']:+.2f}, {shift['hi']:+.2f}])"
    _axes(ax, f"Surplus by model pairing (RQ5){sub}", "", "Seller surplus share (95% CI)")
    colors = {strong: SERIES[0], weak: SERIES[1]}
    for i, k in enumerate(order):
        s = summ[k]
        ax.bar(i, s["surplus_share"], width=0.6, color=colors[k[0]], yerr=_err(s, "surplus_share"), error_kw={"ecolor": INK, "elinewidth": 1},
               label=f"seller = {k[0]}")
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=9, loc="lower right")
    ax.set_xticks(range(len(order)), [f"{a} sells\nto {b}" for a, b in order], fontsize=9)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("even split", (len(order) - 0.5, 0.5), xytext=(0, 4), textcoords="offset points", color=MUTED, fontsize=9, ha="right")
    fig.tight_layout()
    return fig


def cost_quality_chart(rows: list[dict], group_by=("seller_agent", "seller_model")):
    """RQ2: cost per game vs seller surplus, with the Pareto frontier (best quality per budget)."""
    pts = [s for s in summarize(rows, group_by=group_by) if s["surplus_share"] is not None]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    _axes(ax, "Cost vs deal quality (RQ2)", "Cost per game (USD)", "Seller surplus share (95% CI)")
    for i, s in enumerate(pts):
        name = " / ".join(str(s[k]) for k in group_by)
        ax.errorbar(s["cost_per_game_usd"], s["surplus_share"], yerr=_err(s, "surplus_share"), fmt="o", ms=8,
                    color=SERIES[0], elinewidth=1.5)
        ax.annotate(name, (s["cost_per_game_usd"], s["surplus_share"]), xytext=(8, 4), textcoords="offset points", fontsize=9, color=INK)
    front, best = [], float("-inf")
    for s in sorted(pts, key=lambda s: (s["cost_per_game_usd"], -s["surplus_share"])):
        if s["surplus_share"] > best:
            front.append(s)
            best = s["surplus_share"]
    if len(front) > 1:
        ax.step([s["cost_per_game_usd"] for s in front], [s["surplus_share"] for s in front], where="post",
                color=SERIES[1], linewidth=2, label="Pareto frontier")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


def calibration_chart(rows: list[dict]):
    """Self-play: seller surplus share per model with CI; 0.5 is the unbiased reference."""
    cal = [c for c in self_play_calibration(rows) if c["surplus_share"] is not None]
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(cal) + 1.8))
    _axes(ax, "Self-play calibration", "Seller surplus share when both sides run the same model (95% CI)")
    for i, c in enumerate(cal):
        ax.errorbar(c["surplus_share"], i, xerr=[[c["surplus_share"] - c["surplus_share_lo"]], [c["surplus_share_hi"] - c["surplus_share"]]],
                    fmt="o", ms=8, color=SERIES[0], elinewidth=1.5)
    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle="--")
    ax.set_yticks(range(len(cal)), [c["model"] for c in cal])
    fig.tight_layout()
    return fig


def save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path
