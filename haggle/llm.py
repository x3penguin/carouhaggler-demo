"""Thin LLM client with token, latency and cost logging (brief §3).

Currently only OpenAI-compatible endpoints (Ollama, vLLM, OpenAI). The Anthropic
provider is added in Phase 1 behind the same `chat()` signature.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI


@dataclass
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    cost_usd: float


class LLMClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str,
        temperature: float,
        max_tokens: int = 300,
        seed: int | None = None,
        price_in_per_mtok: float = 0.0,
        price_out_per_mtok: float = 0.0,
        log_path: str | Path | None = None,
        provider: str = "openai_compatible",
    ):
        if provider != "openai_compatible":
            raise NotImplementedError(f"Provider '{provider}' is not implemented yet.")
        self.model = model
        self.temperature = temperature
        # Hard cap: constrained decoding on small models can otherwise loop on whitespace forever.
        self.max_tokens = max_tokens
        self.seed = seed
        self.price_in = price_in_per_mtok
        self.price_out = price_out_per_mtok
        self.log_path = Path(log_path) if log_path else None
        # The SDK insists on a key even for Ollama, which ignores it.
        self._client = OpenAI(base_url=base_url, api_key=os.getenv(api_key_env) or "unused")

    def chat(self, messages: list[dict], json_schema: dict | None = None) -> LLMResponse:
        """One chat completion. With json_schema, decoding is constrained to that schema,
        which is what keeps small local models from producing unparseable output."""
        kwargs = {}
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": json_schema, "strict": True},
            }
        start = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            **kwargs,
        )
        latency = time.perf_counter() - start

        usage = resp.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        result = LLMResponse(
            text=resp.choices[0].message.content or "",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_s=latency,
            cost_usd=(tokens_in * self.price_in + tokens_out * self.price_out) / 1_000_000,
        )
        self._log(messages, result)
        return result

    def _log(self, messages: list[dict], result: LLMResponse) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"model": self.model, "messages": messages, **result.__dict__}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
