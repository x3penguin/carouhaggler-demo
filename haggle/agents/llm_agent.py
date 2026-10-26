"""Shared machinery for single-prompt LLM agents (Agent A and the LLM buyers).

Turns an Observation into chat messages, calls the model with the Action JSON
schema, and parses the result. Subclasses only supply the prompt variables.
"""

from pathlib import Path
from string import Template

from pydantic import BaseModel, ValidationError

from haggle import guardrails
from haggle.agents.base import AgentReply, Observation, PublicTurn
from haggle.llm import LLMClient
from haggle.schemas import Action

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
ACTION_SCHEMA = Action.model_json_schema()
USAGE_KEYS = ("tokens_in", "tokens_out", "latency_s", "cost_usd")


def money(x: float) -> str:
    return f"{x:.0f}" if float(x).is_integer() else f"{x:.2f}"


def load_prompt(name: str) -> Template:
    return Template((PROMPT_DIR / name).read_text(encoding="utf-8"))


def describe_turn(turn: PublicTurn, wrap: bool = False) -> str:
    """How the opponent's turn appears to the agent: their words plus the structured move.

    With wrap=True the words are tagged as untrusted (injection defence); the structured
    move stays outside the tags because the engine, not the opponent, produced it.
    """
    if turn.action == "offer":
        move = f"offer ${money(turn.price)}"
    elif turn.action in ("reject", "chat"):
        move = "chat / message" if turn.message else "stay silent / left on read"
    else:
        move = turn.action
    delay_note = f" (after waiting {turn.wait_hours}h)" if turn.wait_hours > 0 else ""
    body = turn.message if turn.message else "(no message sent)"
    if wrap:
        body = guardrails.wrap_untrusted(body, turn.speaker)
    return f"{body}\n\n[{turn.speaker.capitalize()}'s action: {move}{delay_note}]"


class PromptedAgent:
    prompt_file: str = ""

    def __init__(self, client: LLMClient, name: str, injection_defense: bool = False):
        self.client = client
        self.name = name
        self.model = client.model
        self.injection_defense = injection_defense
        self.template = load_prompt(self.prompt_file)

    def prompt_vars(self, obs: Observation) -> dict:
        raise NotImplementedError

    def system_prompt(self, obs: Observation) -> str:
        text = self.template.substitute(self.prompt_vars(obs))
        if self.injection_defense:
            text += guardrails.untrusted_note("buyer" if obs.role == "seller" else "seller")
        return text

    def status_lines(self, obs: Observation) -> list[str]:
        """Engine-provided facts shown after the latest message. Subclasses may add lines."""
        theirs = "none yet" if obs.their_standing_offer is None else f"${money(obs.their_standing_offer)}"
        mine = "none yet" if obs.my_standing_offer is None else f"${money(obs.my_standing_offer)}"
        lines = [f"[Status: your current offer {mine}; their current offer {theirs}]"]
        if obs.deadline_hours is not None:
            days_left = obs.hours_remaining / 24.0
            lines.append(
                f"[Timeline: Day {obs.current_time_hours / 24.0:.1f} | {days_left:.1f} days "
                f"({obs.hours_remaining:.0f}h) left before your private deadline]"
            )
        if obs.last_delay_hours > 0:
            lines.append(f"[Pacing: the other party waited {obs.last_delay_hours:.0f} hours before replying]")
        if obs.highest_competing_offer is not None:
            # Labelled as verified so agents can tell it apart from claims made in chat.
            lines.append(
                f"[Platform-verified: another buyer's current offer is ${money(obs.highest_competing_offer)}; "
                f"{obs.active_inquirers_count} buyers are chatting about this item]"
            )
        if obs.feedback:
            lines.append(f"[Platform: {obs.feedback}]")
        return lines

    def messages(self, obs: Observation, extra_lines: list[str] | None = None) -> list[dict]:
        """Chat history from this agent's point of view. extra_lines are appended after the
        status (Agent B uses this for its private plan, tool results and notes)."""
        msgs = [{"role": "system", "content": self.system_prompt(obs)}]
        if not obs.transcript or obs.transcript[0].speaker == obs.role:
            msgs.append({"role": "user", "content": "(The chat has just opened. Send your first message.)"})
        for turn in obs.transcript:
            if turn.speaker == obs.role:
                own = Action(message=turn.message, action=turn.action, price=turn.price, wait_hours=turn.wait_hours)
                msgs.append({"role": "assistant", "content": own.model_dump_json()})
            else:
                msgs.append({"role": "user", "content": describe_turn(turn, wrap=self.injection_defense)})
        msgs[-1]["content"] += "\n\n" + "\n".join(self.status_lines(obs) + (extra_lines or []))
        return msgs

    def call_json(self, messages: list[dict], model: type[BaseModel]) -> tuple[BaseModel | None, dict]:
        """One schema-constrained call. Returns (parsed or None, usage dict)."""
        resp = self.client.chat(messages, json_schema=model.model_json_schema())
        usage = {k: getattr(resp, k) for k in USAGE_KEYS}
        try:
            return model.model_validate_json(resp.text), usage
        except ValidationError:
            return None, usage

    def act(self, obs: Observation) -> AgentReply:
        action, usage = self.call_json(self.messages(obs), Action)
        if action is None:
            return AgentReply(action=None, error="Reply must be JSON matching the Action schema.", **usage)
        return AgentReply(action=action, **usage)
