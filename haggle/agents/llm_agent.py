"""Shared machinery for single-prompt LLM agents (Agent A and the LLM buyers).

Turns an Observation into chat messages, calls the model with the Action JSON
schema, and parses the result. Subclasses only supply the prompt variables.
"""

from pathlib import Path
from string import Template

from pydantic import ValidationError

from haggle.agents.base import AgentReply, Observation, PublicTurn
from haggle.llm import LLMClient
from haggle.schemas import Action

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
ACTION_SCHEMA = Action.model_json_schema()


def money(x: float) -> str:
    return f"{x:.0f}" if float(x).is_integer() else f"{x:.2f}"


def describe_turn(turn: PublicTurn) -> str:
    """How the opponent's turn appears to the agent: their words plus the structured move."""
    if turn.action == "offer":
        move = f"offer ${money(turn.price)}"
    elif turn.action in ("reject", "chat"):
        if not turn.message:
            move = "stay silent / left on read"
        else:
            move = "chat / message"
    else:
        move = turn.action
    delay_note = f" (after waiting {turn.wait_hours}h)" if turn.wait_hours > 0 else ""
    msg_body = turn.message if turn.message else "(no message sent)"
    return f"{msg_body}\n\n[{turn.speaker.capitalize()}'s action: {move}{delay_note}]"


class PromptedAgent:
    prompt_file: str = ""

    def __init__(self, client: LLMClient, name: str):
        self.client = client
        self.name = name
        self.model = client.model
        self.template = Template((PROMPT_DIR / self.prompt_file).read_text(encoding="utf-8"))

    def prompt_vars(self, obs: Observation) -> dict:
        raise NotImplementedError

    def messages(self, obs: Observation) -> list[dict]:
        msgs = [{"role": "system", "content": self.template.substitute(self.prompt_vars(obs))}]
        if not obs.transcript or obs.transcript[0].speaker == obs.role:
            msgs.append({"role": "user", "content": "(The chat has just opened. Send your first message.)"})
        for turn in obs.transcript:
            if turn.speaker == obs.role:
                own = Action(message=turn.message, action=turn.action, price=turn.price)
                msgs.append({"role": "assistant", "content": own.model_dump_json()})
            else:
                msgs.append({"role": "user", "content": describe_turn(turn)})

        theirs = "none yet" if obs.their_standing_offer is None else f"${money(obs.their_standing_offer)}"
        mine = "none yet" if obs.my_standing_offer is None else f"${money(obs.my_standing_offer)}"
        status = f"[Status: your current offer {mine}; their current offer {theirs}]"
        if obs.deadline_hours is not None:
            days_left = obs.hours_remaining / 24.0
            status += f"\n[Timeline: Day {obs.current_time_hours / 24.0:.1f} | {days_left:.1f} days ({obs.hours_remaining:.0f}h) remaining until your private deadline]"
        if obs.last_delay_hours > 0:
            status += f"\n[Pacing: The other party delayed {obs.last_delay_hours:.0f} hours before responding]"
        if obs.highest_competing_offer is not None:
            status += f"\n[Market Alert: Another buyer made an outside offer of ${money(obs.highest_competing_offer)}]"
        if obs.feedback:
            status += f"\n[Platform: {obs.feedback}]"
        msgs[-1]["content"] += "\n\n" + status
        return msgs

    def act(self, obs: Observation) -> AgentReply:
        resp = self.client.chat(self.messages(obs), json_schema=ACTION_SCHEMA)
        usage = dict(
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            latency_s=resp.latency_s,
            cost_usd=resp.cost_usd,
        )
        try:
            action = Action.model_validate_json(resp.text)
        except ValidationError as e:
            return AgentReply(action=None, error=f"Reply must be JSON matching the schema ({e.error_count()} errors).", **usage)
        return AgentReply(action=action, **usage)
