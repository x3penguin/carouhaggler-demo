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
cp .env.example .env         # only needed for hosted models (set OPENROUTER_API_KEY)
```

## Run

```bash
uv run streamlit run app/demo.py     # demo UI at http://localhost:8501
uv run pytest                        # engine and metric tests
```

`uv run pytest` includes the sanity-agent tests: rule-based agents play full games and the
metrics must return their known values (e.g. surplus share 1.0 and 0.0) before any LLM result
is trusted. Hosted models go through OpenRouter: add a `models:` entry with `base_url:
https://openrouter.ai/api/v1` (example in `configs/demo.yaml`). Not every model honours the `seed`
parameter, so replays against hosted models may not be bit-identical.

Each app session writes to `runs/demo/<timestamp>/`: `games.jsonl` (one `GameResult` per finished
game), `llm_calls.jsonl` (every model call with tokens and latency) and a copy of the config.

## Layout

| Path | What |
|---|---|
| `haggle/schemas.py` | Pydantic models: `Listing`, `Scenario`, `Action`, `Turn`, `GameResult` |
| `haggle/engine.py` | Turn loop and rules: standing offers, accept/offer/reject/walk away, retries, guardrails |
| `haggle/llm.py` | OpenAI-compatible client (Ollama, OpenRouter) with JSON-schema output, retries, usage logging and budget cap |
| `haggle/agents/` | `prompted_seller.py` (Agent A), `agentic.py` (Agent B + agentic buyer), `buyers.py` (LLM personas), `attackers.py` (red-team buyers + twins), shared `llm_agent.py` |
| `haggle/guardrails.py` | Agent-side guardrails: injection wrapping, grounding check, floor-leak filter (`GuardedAgent`) |
| `haggle/prompts/` | Prompt templates |
| `haggle/agents/sanity.py` | Rule-based test agents with known outcomes (test-only) |
| `haggle/tools/` | `price_lookup` (BM25 over comparables), calculator, walk-away estimator |
| `haggle/scenarios.py` | Seeded floor/budget generation and scenario batches |
| `haggle/metrics.py` | Surplus share, below-floor deals, floor leak, guardrail counts |
| `configs/demo.yaml` | All model names, seeds and settings |
| `data/listings_sg.json` | 30 listings, each with at least one flaw |
| `data/attacks.json` | 50 attack scripts in 7 categories; each has a plain twin generated in code |
| `data/comparables.json` | Price comparables for `price_lookup` (placeholder until real data is added, see `data/README.md`) |
| `docs/INTERFACES.md` | Interface contract for teammates |

## Agents (Role 2)

Every agent has `name`, `model` and `act(observation) -> AgentReply`, so the engine, the
orchestrator and the runner treat them the same way. `client` comes from `config.make_client`.

```python
from haggle.agents.prompted_seller import PromptedSeller          # Agent A
from haggle.agents.agentic import AgenticSeller, AgenticBuyer      # Agent B, agentic buyer
from haggle.agents.buyers import LLMBuyer, PERSONAS, BUYER_DEADLINES
from haggle.agents.attackers import AttackBuyer, load_attacks, SUCCESS
from haggle.guardrails import GuardedAgent

PromptedSeller(client, honesty_instruction=True, injection_defense=False)
AgenticSeller(client, use_tools=True, use_planning=True, use_reflection=True)  # ablations for RQ1
LLMBuyer(client, persona="tough", probe_condition=True)  # fair, tough, lowballer, manipulator, indecisive
AttackBuyer(script, twin=False, client=None)  # twin=True: same prices, neutral words; client: LLM rewording
GuardedAgent(seller, leak_filter=True, grounding="block")  # "off" | "flag" | "block"
```

- **Ablation names** come out as e.g. `agentic_seller[tools+plan]`, so results group cleanly.
- **Attack effect** = success rate under `AttackBuyer(script)` minus under
  `AttackBuyer(script, twin=True)` on the same scenarios. `SUCCESS[category]` says whether
  success means a below-floor deal, a floor leak, or either.
- **Buyer urgency**: set `Scenario.buyer_deadline_days` from `BUYER_DEADLINES`
  (urgent 1.5, normal 5, relaxed 10 days).
- **Guardrail events** from `GuardedAgent` land in `Turn.guardrail_events` next to the
  engine's own.

## Open decisions

- Frontier model (reference and judge), budget model (opponents), 1–3B model to fine-tune
- Total API budget
- Live demo vs recorded video
