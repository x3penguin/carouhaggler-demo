"""Thin LLM client with token, latency and cost logging (brief §3).

Only OpenAI-compatible endpoints (Ollama, vLLM, OpenAI, OpenRouter) are supported; the
team uses OpenRouter for hosted models, which speaks the same protocol. `provider` stays as
the extension point if a native client is ever needed.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import openai
from openai import OpenAI


class BudgetExceeded(RuntimeError):
    """Raised once cumulative spend passes max_budget_usd. The runner should catch it and stop."""


class CostTracker:
    """Shared across clients so the budget caps the whole experiment, not one model."""

    def __init__(self, max_budget_usd: float | None = None):
        self.max_budget_usd = max_budget_usd
        self.spent_usd = 0.0

    def add(self, cost_usd: float) -> None:
        self.spent_usd += cost_usd
        if self.max_budget_usd is not None and self.spent_usd > self.max_budget_usd:
            raise BudgetExceeded(
                f"Spent ${self.spent_usd:.4f}, over the ${self.max_budget_usd:.4f} budget."
            )


def _is_transient(e: Exception) -> bool:
    """Timeouts, dropped connections, rate limits and server errors are worth retrying."""
    if isinstance(e, (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError)):
        return True
    return isinstance(e, openai.APIStatusError) and e.status_code >= 500


@dataclass
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    cost_usd: float
    call_id: str = ""


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
        tracker: CostTracker | None = None,
        max_attempts: int = 4,
        backoff_s: float = 1.0,
        sleep=time.sleep,
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
        self.tracker = tracker
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self._sleep = sleep  # injectable so tests don't actually wait
        # The SDK insists on a key even for Ollama, which ignores it.
        # max_retries=0: we do our own retry below so attempts are bounded and testable.
        self._client = OpenAI(
            base_url=base_url, api_key=os.getenv(api_key_env) or "unused", max_retries=0
        )

    def _create_with_retry(self, **kwargs):
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as e:
                if attempt == self.max_attempts or not _is_transient(e):
                    raise
                self._sleep(self.backoff_s * 2 ** (attempt - 1))

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
        resp = self._create_with_retry(
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
            call_id=uuid.uuid4().hex[:12],
        )
        self._log(messages, result)
        if self.tracker is not None:
            self.tracker.add(result.cost_usd)  # after logging, so the call that broke budget is kept
        return result

    def _log(self, messages: list[dict], result: LLMResponse) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "messages": messages,
            **result.__dict__,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
