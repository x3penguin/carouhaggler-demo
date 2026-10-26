"""LLM client tests with a fake SDK client: no network, no Ollama."""

import json
from types import SimpleNamespace

import openai
import pytest

from haggle.llm import BudgetExceeded, CostTracker, LLMClient


def fake_response(text="{}", tin=1000, tout=500):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=tin, completion_tokens=tout),
    )


def make(responses, **kw):
    """Client whose SDK returns/raises the given items in order."""
    client = LLMClient(
        model="m", base_url="http://x", api_key_env="NOPE", temperature=0.0,
        sleep=lambda s: None, **kw,
    )
    items = list(responses)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        item = items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.calls = calls
    return client


def timeout():
    # Built without __init__ so the tests don't depend on the SDK's HTTP library internals.
    return openai.APITimeoutError.__new__(openai.APITimeoutError)


def status_error(code):
    e = openai.APIStatusError.__new__(openai.APIStatusError)
    e.status_code = code
    return e


def test_cost_computed_from_prices():
    c = make([fake_response(tin=1_000_000, tout=500_000)], price_in_per_mtok=2.0, price_out_per_mtok=4.0)
    r = c.chat([{"role": "user", "content": "hi"}])
    assert r.cost_usd == pytest.approx(2.0 + 2.0)
    assert r.tokens_in == 1_000_000 and r.call_id


def test_retries_transient_errors_then_succeeds():
    c = make([timeout(), status_error(503), fake_response("ok")])
    assert c.chat([]).text == "ok" and len(c.calls) == 3


def test_retry_stops_at_cap():
    c = make([timeout()] * 3, max_attempts=3)
    with pytest.raises(openai.APITimeoutError):
        c.chat([])
    assert len(c.calls) == 3


def test_non_transient_error_is_not_retried():
    c = make([status_error(400), fake_response()])
    with pytest.raises(openai.APIStatusError):
        c.chat([])
    assert len(c.calls) == 1


def test_budget_exceeded_raises_after_logging(tmp_path):
    tracker = CostTracker(max_budget_usd=1.0)
    log = tmp_path / "calls.jsonl"
    c = make(
        [fake_response(tin=1_000_000, tout=0)] * 2,
        price_in_per_mtok=0.6, tracker=tracker, log_path=log,
    )
    c.chat([])  # 0.6 spent, under budget
    with pytest.raises(BudgetExceeded):
        c.chat([])  # 1.2 spent
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(lines) == 2 and "timestamp" in lines[0] and "call_id" in lines[0]


def test_json_schema_passed_as_response_format():
    c = make([fake_response()])
    c.chat([], json_schema={"type": "object"})
    assert c.calls[0]["response_format"]["json_schema"]["schema"] == {"type": "object"}
