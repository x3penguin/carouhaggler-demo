"""The agentic loop shared by Agent B (seller) and the agentic buyer: reflect → tools → plan → act.

Each stage is a switch, so RQ1 can be answered by ablation (A, A+tools, A+planning, full B)
instead of only "B beats A". All variants use the same base prompt as Agent A, so the only
difference between them is the scaffolding.

Design choices worth knowing:
- Tools run deterministically in code every turn; the model decides what to do with the
  results, not whether to call them. Small models are unreliable at function calling.
- Reflection happens at the start of a turn, so it can look at how the opponent reacted.
- The plan is cached per turn: engine retries (bad JSON, guardrail re-asks) reuse it
  instead of paying for planning again. A retry is recognised by its feedback, which the
  engine and GuardedAgent always set; a fresh turn never has any.
- Memory is kept per chat (session_id) and reset when a new game starts, so notes never
  leak between buyers or between games.
"""

from dataclasses import dataclass, field

from pydantic import BaseModel

from haggle.agents.base import AgentReply, Observation
from haggle.agents.buyers import LLMBuyer
from haggle.agents.llm_agent import USAGE_KEYS, load_prompt, money
from haggle.agents.prompted_seller import PromptedSeller
from haggle.schemas import Action
from haggle.tools import call_tool


class Plan(BaseModel):
    estimate_of_their_limit: float | None
    strategy: str
    target_price: float | None
    next_move: str


class Reflection(BaseModel):
    lesson: str


@dataclass
class _TurnPlan:
    lines: list[str]  # shown to the model in the act step
    notes: str  # logged as Turn.private_notes
    tool_calls: list[dict]


@dataclass
class _Memory:
    lessons: list[str] = field(default_factory=list)
    market: dict | None = None  # price_lookup result, fetched once per chat
    plans: dict[int, _TurnPlan] = field(default_factory=dict)


class AgenticLoop:
    """Mixin for a PromptedAgent subclass. Call setup_loop() from __init__."""

    other_side = ""  # "buyer" or "seller"
    their_limit = ""  # how the opponent's private number is described in prompts
    max_lessons = 3

    def setup_loop(self, use_tools: bool, use_planning: bool, use_reflection: bool) -> None:
        self.use_tools = use_tools
        self.use_planning = use_planning
        self.use_reflection = use_reflection
        self.plan_template = load_prompt("agentic_plan.md")
        self.reflect_template = load_prompt("agentic_reflect.md")
        self._memories: dict[tuple, _Memory] = {}

    def variant_name(self, base: str) -> str:
        parts = [p for p, on in (("tools", self.use_tools), ("plan", self.use_planning),
                                 ("reflect", self.use_reflection)) if on]
        return f"{base}[{'+'.join(parts) or 'none'}]"

    # ---- memory ----------------------------------------------------------------

    def _memory(self, obs: Observation) -> _Memory:
        key = (obs.session_id, obs.listing.id)
        first_turn = not any(t.speaker == obs.role for t in obs.transcript)
        if first_turn and obs.feedback is None:
            self._memories[key] = _Memory()  # a new game in this chat: start fresh
        return self._memories.setdefault(key, _Memory())

    # ---- tools (role-specific) -------------------------------------------------

    def run_tools(self, obs: Observation, mem: _Memory) -> tuple[list[str], list[dict]]:
        calls = []
        if mem.market is None:
            call = call_tool("price_lookup", {"query": obs.listing.title, "category": obs.listing.category})
            mem.market = call["result"]
            calls.append(call)
        median = mem.market.get("median") if isinstance(mem.market, dict) else None
        lines = []
        if median is not None:
            examples = "; ".join(f"{e.get('title')} ({e.get('condition') or 'condition n/a'}) ${money(e['price_sgd'])}"
                                 for e in mem.market.get("examples", []) if e.get("price_sgd") is not None)
            lines.append(f"Market check: median comparable price ${money(median)} "
                         f"from {mem.market.get('n', 0)} comparables. Closest: {examples}")
        else:
            lines.append("Market check: no comparable prices found.")
        return lines, calls

    # ---- the loop --------------------------------------------------------------

    def _reflect(self, obs: Observation, mem: _Memory, usage: dict) -> None:
        own = [i for i, t in enumerate(obs.transcript) if t.speaker == obs.role]
        if not own or own[-1] == len(obs.transcript) - 1:
            return  # nothing new from the opponent to learn from yet
        msgs = self.messages(obs, extra_lines=["", self.reflect_template.substitute(
            other=self.other_side, their_limit=self.their_limit)])
        parsed, u = self.call_json(msgs, Reflection)
        _add(usage, u)
        if parsed is not None and parsed.lesson.strip():
            mem.lessons = (mem.lessons + [parsed.lesson.strip()])[-self.max_lessons:]

    def _plan_turn(self, obs: Observation, mem: _Memory, usage: dict) -> _TurnPlan:
        if self.use_reflection:
            self._reflect(obs, mem, usage)
        tool_lines, tool_calls = self.run_tools(obs, mem) if self.use_tools else ([], [])
        context = [f"[Tool] {line}" for line in tool_lines]
        if mem.lessons:
            context.append("[Your notes from earlier turns] " + " | ".join(mem.lessons))

        plan = None
        if self.use_planning:
            msgs = self.messages(obs, extra_lines=["", self.plan_template.substitute(
                other=self.other_side, their_limit=self.their_limit,
                context="\n".join(context) + ("\n" if context else ""))])
            plan, u = self.call_json(msgs, Plan)
            _add(usage, u)

        lines = list(context)
        if plan is not None:
            estimate = "unknown" if plan.estimate_of_their_limit is None else f"${money(plan.estimate_of_their_limit)}"
            target = "none" if plan.target_price is None else f"${money(plan.target_price)}"
            lines.append(f"[Your private plan] Estimated {self.their_limit}: {estimate}. Strategy: {plan.strategy} "
                         f"Target price: {target}. Next move: {plan.next_move}")
        notes = "\n".join(lines) or None
        return _TurnPlan(lines=lines, notes=notes, tool_calls=tool_calls)

    def act(self, obs: Observation) -> AgentReply:
        mem = self._memory(obs)
        usage = {k: 0 for k in USAGE_KEYS}
        turn_key = len(obs.transcript)
        if obs.feedback is None or turn_key not in mem.plans:
            mem.plans[turn_key] = self._plan_turn(obs, mem, usage)
        turn_plan = mem.plans[turn_key]

        action, u = self.call_json(self.messages(obs, extra_lines=turn_plan.lines), Action)
        _add(usage, u)
        if action is None:
            return AgentReply(action=None, error="Reply must be JSON matching the Action schema.", **usage)
        return AgentReply(action=action, private_notes=turn_plan.notes, tool_calls=turn_plan.tool_calls, **usage)


def _add(total: dict, usage: dict) -> None:
    for k in USAGE_KEYS:
        total[k] += usage[k]


class AgenticSeller(AgenticLoop, PromptedSeller):
    """Agent B. With all three switches off it behaves exactly like Agent A."""

    other_side = "buyer"
    their_limit = "maximum budget"

    def __init__(self, client, use_tools: bool = True, use_planning: bool = True,
                 use_reflection: bool = True, honesty_instruction: bool = True,
                 injection_defense: bool = False, name: str | None = None):
        PromptedSeller.__init__(self, client, name="agentic_seller", honesty_instruction=honesty_instruction,
                                injection_defense=injection_defense)
        self.setup_loop(use_tools, use_planning, use_reflection)
        self.name = name or self.variant_name("agentic_seller")

    def run_tools(self, obs: Observation, mem: _Memory) -> tuple[list[str], list[dict]]:
        lines, calls = super().run_tools(obs, mem)
        buyer_offers = [t.price for t in obs.transcript if t.speaker == "buyer" and t.price is not None]
        days = obs.hours_remaining / 24.0 if obs.hours_remaining is not None else None
        median = mem.market.get("median") if isinstance(mem.market, dict) else None

        wav = call_tool("walk_away_value", {
            "floor": obs.private_limit, "buyer_offers": buyer_offers, "market_median": median,
            "turns_left": obs.turns_left, "days_remaining": days,
            "highest_competing_offer": obs.highest_competing_offer,
        })
        calls.append(wav)
        if "reason" in wav["result"]:
            lines.append(f"Walk-away estimate: {wav['result']['reason']}")

        current = obs.my_standing_offer if obs.my_standing_offer is not None else obs.listing.asking_price
        conc = call_tool("calculator", {"op": "next_concession", "current": current,
                                        "floor": obs.private_limit, "remaining_turns": max(obs.turns_left, 1)})
        calls.append(conc)
        if isinstance(conc["result"], (int, float)):
            lines.append(f"If you concede evenly to your floor, your next price would be ${money(conc['result'])}.")
        return lines, calls


class AgenticBuyer(AgenticLoop, LLMBuyer):
    """Mirror of Agent B for buyers: same loop, buyer prompt, market check only."""

    other_side = "seller"
    their_limit = "minimum acceptable price"

    def __init__(self, client, persona: str = "fair", use_tools: bool = True, use_planning: bool = True,
                 use_reflection: bool = True, probe_condition: bool = True,
                 injection_defense: bool = False, name: str | None = None):
        LLMBuyer.__init__(self, client, persona=persona, probe_condition=probe_condition,
                          injection_defense=injection_defense)
        self.setup_loop(use_tools, use_planning, use_reflection)
        self.name = name or self.variant_name(f"agentic_buyer:{persona}")
