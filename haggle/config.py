"""YAML config loading. Model names, prices and paths live in configs/, never in code."""

import hashlib
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

from haggle.llm import CostTracker, LLMClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    with open(PROJECT_ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    """Short, stable fingerprint so every logged game can be traced to its exact config."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def resolve(path: str) -> Path:
    return PROJECT_ROOT / path


def make_client(
    cfg: dict, model_key: str, log_path: Path | None = None, tracker: CostTracker | None = None
) -> LLMClient:
    m = cfg["models"][model_key]
    return LLMClient(
        provider=m["provider"],
        model=m["model"],
        base_url=m["base_url"],
        api_key_env=m["api_key_env"],
        temperature=m["temperature"],
        max_tokens=m["max_tokens"],
        seed=cfg["seed"],
        price_in_per_mtok=m["price_in_per_mtok"],
        price_out_per_mtok=m["price_out_per_mtok"],
        log_path=log_path,
        tracker=tracker,
        max_attempts=m.get("max_attempts", 4),
        backoff_s=m.get("backoff_s", 1.0),
    )
