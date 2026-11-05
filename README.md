# Haggle Arena

A negotiation testbed for evaluating LLM seller agents on second-hand marketplace listings
(Singapore-style). It's a pre-deployment safety check: does the agent get good deals, can buyers
manipulate it, does it lie, and does the stronger model win? See `CLAUDE.md` for the full brief.

**Status:** engine, agents, evaluation and all experiment configs are built. The fine-tuning pipeline
is ready but the model isn't trained yet: see `finetune/HANDOVER.md`. Choose the hosted models in
`configs/models.yaml` before running experiments.

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
| `configs/models.yaml` | The models and prices for every experiment |
| `configs/*.yaml` | One config per experiment; `demo.yaml` is the local demo |
| `haggle/assist.py` | RQ6: draft, then human approve or edit, then deliver |
| `finetune/` | RQ2: CraigslistBargain conversion, QLoRA training (MLX or Colab), Ollama export, tuned-vs-stock check. Start at `HANDOVER.md` |
| `data/listings_sg.json` | 104 listings, each with at least one flaw |
| `data/attacks.json` | 50 attack scripts in 7 categories; each has a plain twin generated in code |
| `data/comparables.json` | 416 price comparables for `price_lookup` (sources in `data/README.md`) |

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

## Evaluation (Role 3)

Fill in `configs/models.yaml` first (model ids and real prices) and add `OPENROUTER_API_KEY` to `.env`.
The runner won't start on a placeholder or a hosted model priced at $0. Add `--dry-run` to any
command to see its game count without playing anything. Run in this order:

```bash
uv run python -m haggle.runner configs/sanity.yaml       # rule-based, no LLM: must pass first
uv run python -m haggle.runner configs/pilot.yaml        # 51 games: gives the real cost per game

# Validate the judge on about 100 hand-labelled games
uv run python -m haggle.judge export runs/pilot/<ts> --out labels/pilot_100.csv --n 100
uv run python -m haggle.judge judge configs/judge.yaml runs/pilot/<ts>
uv run python -m haggle.judge agree labels/pilot_100.csv runs/pilot/<ts>/honesty.jsonl

uv run python -m haggle.runner configs/self_play.yaml    # calibration, 60 games
uv run python -m haggle.runner configs/replay.yaml       # paired replay of the pilot, 60 games
uv run python -m haggle.runner configs/tournament.yaml   # RQ1 ablation, 240 games
uv run python -m haggle.runner configs/honesty.yaml      # RQ4 honesty off, 120 games
uv run python -m haggle.runner configs/attacks.yaml      # RQ3, 400 games
uv run python -m haggle.runner configs/asymmetry.yaml    # RQ5, 120 games (needs Ollama)
uv run python -m haggle.runner configs/opponent.yaml     # RQ5 agentic buyer, 60 games
uv run python -m haggle.runner configs/extension.yaml    # 40 markets, 120 chats
uv run python -m haggle.runner configs/assist.yaml       # RQ6 autopilot arm, 15 games
```

**RQ2 (Agents C and D):** these run on the same scenarios as A and B, once the fine-tuned model
exists. Put its GGUF in `runs/finetune/`, then:

```bash
cp finetune/Modelfile runs/finetune/Modelfile
(cd runs/finetune && ollama create qwen2.5-3b-haggle -f Modelfile)
uv run python -m haggle.runner configs/tournament_finetuned.yaml   # 120 games
uv run python -m haggle.runner configs/attacks_finetuned.yaml      # 400 games, no API key needed
```

**RQ6 human arm:** run the demo and open the **Seller Assist** page, then have each participant play
scenarios from the list. Games are saved to `runs/assist_human/<timestamp>/`.

**Judging:** run `haggle.judge judge` on the tournament and honesty runs (RQ4), and on both RQ6 arms
(`runs/assist/<ts>` and each `runs/assist_human/<ts>`). Budgets in the
configs are safety caps, so review them against the pilot's cost per game.

Each run writes `runs/<experiment>/<timestamp>/` with the resolved config, `games.jsonl`,
`llm_calls.jsonl`, `summary.json`/`.txt` (mean and bootstrap 95% CI per metric) and `manifest.json`
(spend, cost per game, crashed games, budget abort). Crashed games are left out of the statistics.
Agents load by name from the config with their switches, e.g.
`{agent: agentic_seller, model: m, use_tools: false, guard: {leak_filter: true, grounding: block}}`.
`{agent: attacks, twins: true}` expands to every script and its twin; `metrics.attack_effect`
reports attack − twin, paired on script and scenario. The notebook reads the newest run of each
experiment; pin frozen paths in Phase 5. Report: `uv sync --group analysis`, then open
`analysis/results.ipynb`.

## Open decisions

- Main seller, buyer pool and judge models (`configs/models.yaml`)
- Total API budget
- Live demo vs recorded video
