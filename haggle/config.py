"""YAML config loading. Model names, prices and paths live in configs/, never in code."""

import hashlib
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from haggle.llm import CostTracker, LLMClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    with open(PROJECT_ROOT / path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # One shared model list, so "the buyer pool stays the same across all conditions" is
    # true by construction and choosing a model means editing one file. A config's own
    # `models:` entries win over the shared ones.
    if cfg.get("models_file"):
        with open(PROJECT_ROOT / cfg["models_file"], encoding="utf-8") as f:
            shared = yaml.safe_load(f)["models"]
        cfg["models"] = {**shared, **(cfg.get("models") or {})}
    return cfg


def config_hash(cfg: dict) -> str:
    """Short, stable fingerprint so every logged game can be traced to its exact config."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def resolve(path: str) -> Path:
    return PROJECT_ROOT / path


PLACEHOLDER = "SET_ME"


def check_models(cfg: dict, model_keys: set[str]) -> None:
    """Refuse to start with a model that isn't really configured.

    A hosted model at $0 is the dangerous case: cost is computed from the configured
    price, so the budget cap would never trigger while the provider keeps charging.
    """
    problems = []
    for key in sorted(model_keys):
        m = cfg["models"][key]
        hosted = not any(h in m["base_url"] for h in ("localhost", "127.0.0.1"))
        if PLACEHOLDER in str(m["model"]):
            problems.append(f"{key}: choose a model (configs/models.yaml)")
        elif hosted and not m.get("allow_zero_price") and not (m["price_in_per_mtok"] or m["price_out_per_mtok"]):
            problems.append(f"{key}: hosted model has a $0 price, so the budget cap can't work")
        if hosted and not os.getenv(m["api_key_env"]):
            problems.append(f"{key}: {m['api_key_env']} is not set (.env)")
    if problems:
        raise ValueError("Model config not ready:\n  " + "\n  ".join(problems))


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
