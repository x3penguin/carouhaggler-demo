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
    move = f"offer ${money(turn.price)}" if turn.action == "offer" else turn.action
    return f"{turn.message}\n\n[{turn.speaker.capitalize()}'s action: {move}]"


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
        status = f"[Status: your current offer {mine}; their current offer {theirs}; your turns left {obs.turns_left}]"
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
