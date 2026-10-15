# Haggle Arena

A negotiation testbed for evaluating LLM seller agents on second-hand marketplace listings
(Singapore-style). It's a pre-deployment safety check: does the agent get good deals, can buyers
manipulate it, does it lie, and does the stronger model win? See `CLAUDE.md` for the full brief.

**Status:** proof-of-concept demo. Engine, prompted seller (Agent A), LLM buyer personas, floor
guardrail and a Streamlit UI work against a local model. Phase 1 (runner, pilot, full metrics,
Anthropic client) is next.

## Setup

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com).

```bash
ollama pull qwen2.5:3b
uv sync                      # add --system-certs if you get "invalid peer certificate"
cp .env.example .env         # only needed later for paid APIs
```

## Run

```bash
uv run streamlit run app/demo.py     # demo UI at http://localhost:8501
uv run pytest                        # engine and metric tests
```

Each app session writes to `runs/demo/<timestamp>/`: `games.jsonl` (one `GameResult` per finished
game), `llm_calls.jsonl` (every model call with tokens and latency) and a copy of the config.

## Layout

| Path | What |
|---|---|
| `haggle/schemas.py` | Pydantic models: `Listing`, `Scenario`, `Action`, `Turn`, `GameResult` |
| `haggle/engine.py` | Turn loop and rules: standing offers, accept/offer/reject/walk away, retries, guardrails |
| `haggle/llm.py` | OpenAI-compatible client (Ollama) with JSON-schema output and usage logging |
| `haggle/agents/` | `prompted_seller.py` (Agent A), `buyers.py` (LLM personas), shared `llm_agent.py` |
| `haggle/prompts/` | Prompt templates |
| `haggle/scenarios.py` | Seeded floor/budget generation |
| `haggle/metrics.py` | Surplus share, below-floor deals, floor leak, guardrail counts |
| `configs/demo.yaml` | All model names, seeds and settings |
| `data/listings_sg.json` | Starter listings, each with at least one flaw |

## Open decisions

- Frontier model (reference and judge), budget model (opponents), 1–3B model to fine-tune
- Total API budget
- Live demo vs recorded video
