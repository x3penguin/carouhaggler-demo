"""Tuned vs stock model on held-out seller turns, through Ollama (RQ2 training-time metrics).

Run from the repo root, once both models are served:

    uv run python -m finetune.compare_models qwen2.5:3b qwen2.5-3b-haggle --out runs/finetune/report.json

Testing the served model, not the training framework's copy, means the GGUF export and the
Modelfile are checked too, and every training route gets scored by the same code.

It sends no JSON schema, so the valid-JSON rate shows whether the model learned the format itself.
Real games enforce the schema, so this measures learning rather than live format failures.
"""

import argparse
import json
import random
import re
from pathlib import Path
from statistics import median

from pydantic import ValidationError

from haggle.llm import LLMClient
from haggle.schemas import Action

ROOT = Path(__file__).resolve().parents[1]
_FLOOR = re.compile(r"\(your floor\) is SGD ([\d.]+)\.")


def parse_reply(text: str) -> Action | None:
    """The reply as an Action, or None if it isn't exactly the format the agent is told to use."""
    try:
        action = Action.model_validate_json(text.strip())
    except ValidationError:
        return None
    if action.action == "offer" and (action.price is None or action.price <= 0):
        return None
    return action


def score(rows: list[dict], replies: list[str]) -> dict:
    """Compare each reply with the good human seller's move and the floor in its prompt."""
    valid = same_action = below_floor = offers = 0
    price_errors = []
    for row, text in zip(rows, replies):
        reply = parse_reply(text)
        if reply is None:
            continue
        valid += 1
        truth = Action.model_validate_json(row["messages"][-1]["content"])
        same_action += reply.action == truth.action
        if reply.action == "offer":
            offers += 1
            floor = float(_FLOOR.search(row["messages"][0]["content"]).group(1))
            below_floor += reply.price < floor
            if truth.action == "offer":
                price_errors.append(abs(reply.price - truth.price) / truth.price)
    return {
        "examples": len(rows),
        "valid_json_rate": valid / len(rows),
        "same_action_as_human": same_action / valid if valid else None,
        "offers_below_floor": below_floor / offers if offers else None,
        "median_price_error": median(price_errors) if price_errors else None,
    }


def evaluate(model: str, rows: list[dict], base_url: str, seed: int) -> dict:
    client = LLMClient(model=model, base_url=base_url, api_key_env="OLLAMA_API_KEY", temperature=0.0,
                       max_tokens=200, seed=seed)
    replies, tokens_out, seconds = [], 0, 0.0
    for i, row in enumerate(rows):
        response = client.chat(row["messages"][:-1])
        replies.append(response.text)
        tokens_out += response.tokens_out
        seconds += response.latency_s
        print(f"{model}: {i + 1}/{len(rows)}", end="\r", flush=True)
    print()
    return {**score(rows, replies), "tokens_per_second": tokens_out / seconds if seconds else None,
            "seconds_per_reply": seconds / len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", help="Ollama model names, e.g. the stock one, then the tuned one")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "craigslist" / "sft" / "valid.jsonl")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines()]
    rows = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    report = {model: evaluate(model, rows, args.base_url, args.seed) for model in args.models}
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
