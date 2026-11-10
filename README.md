# Haggle Arena

A negotiation testbed for evaluating LLM seller agents on second-hand marketplace listings
(Singapore-style). It's a pre-deployment safety check: does the agent get good deals, can buyers
manipulate it, does it lie, and does the stronger model win? See `CLAUDE.md` for the full brief.

**Status:** engine, agents, evaluation and all experiment configs are built, and the hosted models
are chosen in `configs/models.yaml`. The fine-tuned seller is trained (`finetune/HANDOVER.md`).

**Fine-tuned model used in the report:** `qwen2.5-3b-haggle-q4_k_m.gguf`, SHA-256
`2b57f5c2d5c62ab2a46d2d8a47f94e3b5c935baa71eb27ca1aba78edf0ed6175`, from the checkpoint at iteration
300 (lowest held-out loss). Training stopped at iteration 430 of the planned 1,570, about half a pass
over the data. The file isn't in git: ask the team for it, then register it with
`ollama create qwen2.5-3b-haggle -f Modelfile` from the folder holding it.

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
command to see its game count without playing anything. If a run stops part-way (crash, low memory),
`--resume runs/<experiment>/<timestamp>` keeps its finished games and plays the rest; it refuses a run of a different
config. Run in this order:

```bash
uv run python -m haggle.runner configs/sanity.yaml       # rule-based, no LLM: must pass first
uv run python -m haggle.runner configs/pilot.yaml        # 51 games: gives the real cost per game

uv run python -m haggle.runner configs/self_play.yaml    # calibration, 60 games
uv run python -m haggle.runner configs/replay.yaml       # paired replay of the pilot, 60 games
uv run python -m haggle.runner configs/tournament.yaml   # RQ1 ablation, 240 games
uv run python -m haggle.runner configs/honesty.yaml      # RQ4 honesty off, 120 games
uv run python -m haggle.runner configs/attacks.yaml      # RQ3, 400 games
uv run python -m haggle.runner configs/asymmetry.yaml    # RQ5, 120 games (needs Ollama)
uv run python -m haggle.runner configs/opponent.yaml     # RQ5 agentic buyer, 60 games
uv run python -m haggle.runner configs/extension_solo.yaml  # RQ6 baseline: 40 markets, 1 buyer each
uv run python -m haggle.runner configs/extension.yaml    # RQ6 competition: 40 markets, 120 chats
```

**RQ2 (Agents C and D):** these run on the same scenarios as A and B, once the fine-tuned model
exists. Put its GGUF in `runs/finetune/`, then:

```bash
cp finetune/Modelfile runs/finetune/Modelfile
(cd runs/finetune && ollama create qwen2.5-3b-haggle -f Modelfile)
uv run python -m haggle.runner configs/tournament_finetuned.yaml   # 120 games
uv run python -m haggle.runner configs/attacks_finetuned.yaml      # 400 games, no API key needed
uv run python -m haggle.runner configs/tournament_finetuned_ablation.yaml  # fine-tuned A+tools, A+plan: 120 games
```

**Untuned base model** (`qwen2.5:3b`, the model we fine-tuned): the same games again, so RQ1 covers every
model and RQ2 compares base -> fine-tuned -> Haiku. Local seller, so only the buyers cost money:

```bash
ollama pull qwen2.5:3b
uv run python -m haggle.runner configs/tournament_base.yaml   # 240 games, ~$0.14 of buyer calls
uv run python -m haggle.runner configs/attacks_base.yaml      # 400 games, no API key needed
```

**DPO model (RQ2 v2):** the fine-tuned model trained further with DPO, `qwen2.5-3b-haggle-dpo-q4_k_m.gguf`, SHA-256
`281be40d0c225b7d85c09e5762e05de754f4ddc0364c0e3972073fa226950c9f`. Not in git: ask the team for it. Copy
`finetune/Modelfile` next to it, change `FROM` to the DPO file, then:

```bash
ollama create qwen2.5-3b-haggle-dpo -f Modelfile
uv run python -m haggle.runner configs/tournament_dpo.yaml   # 240 games, ~$0.27 of buyer calls
uv run python -m haggle.runner configs/attacks_dpo.yaml      # 400 games, no API key needed
```

v2 cut below-floor deals, but mostly by no longer closing any. The plan for v3 is `finetune/DPO_V3.md`. Before
training, check a preference file with `uv run python -m finetune.audit_dpo_pairs <train.jsonl>`. Before a paid
tournament, check that a local seller closes deals with `configs/closing_check.yaml` (rule-based buyers, $0).

**DPO v3:** retrained from the SFT model on pairs that also teach closing (`uv run python -m finetune.build_dpo_pairs
--seed 42` rebuilds `data/dpo_v3/`; training in `finetune/dpo_colab.ipynb`). The model is
`qwen2.5-3b-haggle-dpo-v3-q4_k_m.gguf`, SHA-256 `2c98a50508f8574ced40292f42b7aef4387d477e510683abb5a0cca7d1d8240a`.
Register it the same way, as `qwen2.5-3b-haggle-dpo-v3`, then:

```bash
uv run python -m haggle.runner configs/closing_check.yaml    # all four local models, $0
uv run python -m haggle.runner configs/attacks_dpo_v3.yaml   # 400 games, no API key needed
```

v3 closed deals but lost 57 of 200 attacks. **DPO v4** (`finetune/DPO_V3.md` §7) rebuilds the pairs so the floor is the
only thing that decides accept or refuse: `uv run python -m finetune.build_dpo_pairs_v4 --seed 42` writes `data/dpo_v4/`,
`finetune/dpo_colab.ipynb` trains it, and `uv run python -m finetune.check_decisions data/dpo_v4/valid.jsonl --models
<keys>` checks a model's decisions on held-out twins for $0 before the closing check and the attack run.

**Judging:** run the judge on the RQ1, RQ2, RQ3 and RQ4 runs and the frontier pilot, then check it
against 30 hand-labelled games (`labels/hand_labels_30.csv`; rules in `haggle/prompts/judge_honesty.md`):

```bash
uv run python -m haggle.judge judge configs/judge.yaml runs/tournament/<ts>   # and honesty, tournament_finetuned,
                                                                              # attacks, attacks_finetuned, pilot_sol,
                                                                              # tournament_base, tournament_finetuned_ablation,
                                                                              # tournament_dpo, attacks_dpo
cat runs/*/*/honesty.jsonl > labels/all_verdicts.jsonl
uv run python -m haggle.judge agree labels/hand_labels_30.csv labels/all_verdicts.jsonl
```

Budgets in the configs are safety caps, so review them against the pilot's cost per game. The runs
used in the report are committed under `runs/` (see `docs/EXPERIMENTS_RUN_LIST.md`), without the raw
call logs.

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

- Live demo vs recorded video

Decided on 7 Nov 2026: a $6 API budget, so the hosted models are cheap ones (Claude Haiku 5.5 seller,
GPT-6 Luna buyers, MiMo V2.6 Flash judge). The 51-game pilot from 5 Nov with GPT-6.1 Sol as the seller
and GPT-6.1 Terra buyers is kept as the frontier reference for RQ2.
