"""A scripted stand-in for LLMClient so agent tests run without Ollama or an API key."""

import json

from haggle.llm import LLMResponse


class FakeClient:
    """Returns queued replies in order. A reply may be a dict (sent as JSON), a raw string,
    or a function of (messages, schema) for replies that depend on the prompt."""

    def __init__(self, replies=None, model: str = "fake-model", tokens: int = 10):
        self.replies = list(replies or [])
        self.model = model
        self.tokens = tokens
        self.calls: list[dict] = []

    def chat(self, messages, json_schema=None) -> LLMResponse:
        self.calls.append({"messages": messages, "schema": (json_schema or {}).get("title")})
        if not self.replies:
            raise AssertionError("FakeClient ran out of scripted replies")
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(messages, json_schema)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return LLMResponse(text=text, tokens_in=self.tokens, tokens_out=self.tokens, latency_s=0.01, cost_usd=0.0)

    def schemas(self) -> list[str]:
        return [c["schema"] for c in self.calls]


def action(message="", action="chat", price=None, wait_hours=0) -> dict:
    return {"message": message, "action": action, "price": price, "wait_hours": wait_hours}
