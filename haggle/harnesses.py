"""Harnesses that keep cross-agent comparisons valid."""

from __future__ import annotations

import random

from haggle.agents.base import AgentReply, Observation
from haggle.metrics import bootstrap_ci
from haggle.schemas import Action, GameResult


class ReplayBuyer:
    """Replays the buyer side of a recorded game, in order."""

    model = "replay"

    def __init__(self, source: GameResult):
        self.source = source
        self.name = f"replay:{source.buyer_agent}"
        self.script = [t.action for t in source.turns if t.speaker == "buyer"]
        offered = [a.price for a in self.script if a.action == "offer"]
        accepted = source.deal_price if source.outcome == "deal" and source.turns[-1].speaker == "buyer" else None
        self.max_agreed = max([p for p in [accepted, *offered] if p is not None], default=None)

    def act(self, obs: Observation) -> AgentReply:
        i = sum(1 for t in obs.transcript if t.speaker == "buyer")
        if i >= len(self.script):
            return AgentReply(
                action=Action(message="Ok never mind, thanks.", action="walk_away", price=None),
                private_notes="replay: script exhausted -> walk_away",
            )
        action = self.script[i]
        if action.action == "accept":
            theirs = obs.their_standing_offer
            if self.max_agreed is not None and theirs is not None and theirs > self.max_agreed:
                return AgentReply(
                    action=Action(message=action.message, action="offer", price=self.max_agreed),
                    private_notes=f"replay: accept of ${theirs:.2f} > original ${self.max_agreed:.2f} -> offer",
                )
        return AgentReply(action=action, private_notes="replay: verbatim")


def replay_divergence(result: GameResult) -> float:
    """Share of replayed buyer turns that had to be adapted (0 = fully verbatim replay)."""
    notes = [t.private_notes or "" for t in result.turns if t.speaker == "buyer"]
    if not notes:
        return 0.0
    return sum(1 for n in notes if not n.endswith("verbatim")) / len(notes)


def self_play_calibration(rows: list[dict], n_boot: int = 2000, seed: int = 0) -> list[dict]:
    """Per self-play model: deal rate and surplus share with CIs, and offset from 0.5."""
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        if r["seller_model"] == r["buyer_model"]:
            by_model.setdefault(r["seller_model"], []).append(r)
    out = []
    for model, rs in sorted(by_model.items()):
        share, lo, hi, n = bootstrap_ci([r["surplus_share"] for r in rs], n_boot=n_boot, seed=seed)
        deal, dlo, dhi, _ = bootstrap_ci([r["deal"] for r in rs], n_boot=n_boot, seed=seed)
        out.append({
            "model": model,
            "games": len(rs),
            "deal_rate": deal, "deal_rate_lo": dlo, "deal_rate_hi": dhi,
            "surplus_share": share, "surplus_share_lo": lo, "surplus_share_hi": hi, "deals": n,
            "offset_from_half": None if share is None else share - 0.5,
            "centred": None if share is None else lo <= 0.5 <= hi,
        })
    return out


def paired_difference(
    rows: list[dict],
    variant_a: str,
    variant_b: str,
    metric: str = "surplus_share",
    variant_key: str = "seller_agent",
    pair_keys: tuple[str, ...] = ("scenario_id", "buyer_agent"),
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Mean of (A − B) over scenarios both variants played, with a bootstrap CI."""
    index: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        if r[variant_key] in (variant_a, variant_b):
            index.setdefault(tuple(r[k] for k in pair_keys), {})[r[variant_key]] = r
    diffs, dropped = [], 0
    for pair in index.values():
        if variant_a in pair and variant_b in pair:
            a, b = pair[variant_a][metric], pair[variant_b][metric]
            if a is None or b is None:
                dropped += 1
            else:
                diffs.append(float(a) - float(b))
    point, lo, hi, n = bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
    win, wlo, whi, _ = bootstrap_ci([1.0 if d > 0 else 0.5 if d == 0 else 0.0 for d in diffs], n_boot=n_boot, seed=seed)
    return {"metric": metric, "a": variant_a, "b": variant_b, "mean_diff": point, "lo": lo, "hi": hi,
            "win_rate_a": win, "win_rate_a_lo": wlo, "win_rate_a_hi": whi, "pairs": n, "dropped": dropped}


def sample(items: list, k: int, seed: int) -> list:
    """Seeded sample without replacement, preserving order (for picking replay sources)."""
    if k >= len(items):
        return list(items)
    idx = sorted(random.Random(seed).sample(range(len(items)), k))
    return [items[i] for i in idx]
