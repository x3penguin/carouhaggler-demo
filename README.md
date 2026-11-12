# Haggle Arena

A negotiation testbed for LLM seller agents on second-hand marketplace listings (Singapore-style). Marketplaces are
starting to let AI agents haggle for sellers; this project is the **pre-deployment safety evaluation** a platform
would want first. Does the agent get good deals? Can buyers manipulate it? Does it lie about the item? Does whoever
has the stronger model win?

| # | Research question | Main configs |
|---|---|---|
| RQ1 | Does an agentic seller (planning + reflection + price tool) beat a prompted seller? Answered by ablation | `tournament`, `tournament_base`, `tournament_finetuned_ablation` |
| RQ2 | Can a fine-tuned 3B open-weight model compete with a prompted hosted model at a fraction of the cost? | `tournament_finetuned`, `attacks_finetuned`, `tournament_dpo`, `attacks_dpo`, `attacks_dpo_v3` |
| RQ3 | How exploitable are seller agents (sob story, fake urgency, lowball, prompt injection, floor extraction, ...)? Measured against matched twins | `attacks`, `attacks_base`, `attacks_floorguard` |
| RQ4 | Do better negotiators lie more? | `honesty` + the honesty judge |
| RQ5 | When a strong model negotiates against a weak one, how much surplus shifts? | `asymmetry`, `opponent` |
| RQ6 | Do deadlines and competing buyers make sellers easier to exploit? | `extension_solo`, `extension` |

The full brief is `CLAUDE.md`. The runs used in the report are committed under `runs/` and listed in
`docs/EXPERIMENTS_RUN_LIST.md`.

## Contents

1. [Setup](#1-setup)
2. [Variables and configuration](#2-variables-and-configuration)
3. [Quick start: tests and demo](#3-quick-start-tests-and-demo)
4. [Running the experiments](#4-running-the-experiments)
5. [Judging and analysis](#5-judging-and-analysis)
6. [Fine-tuning and DPO](#6-fine-tuning-and-dpo)
7. [Reproducibility](#7-reproducibility)
8. [Repository layout](#8-repository-layout)

## 1. Setup

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com) for the local
models. Hosted models go through [OpenRouter](https://openrouter.ai), which needs an API key.

```bash
git clone <this repo> && cd carouhaggler-demo
uv sync                          # core dependencies; add --system-certs if you get "invalid peer certificate"
uv sync --group analysis         # optional: Jupyter + matplotlib for analysis/results.ipynb
cp .env.example .env             # then fill in OPENROUTER_API_KEY (see section 2)

ollama pull qwen2.5:3b           # the untuned base model: RQ5's weak side and the RQ2 baseline
ollama pull llama3.2:3b          # optional: only for the demo's honesty-judge button
```

**Fine-tuned models (RQ2).** These aren't in git. **[Download all four GGUFs here](https://drive.google.com/drive/folders/1aKUYzpfbQcsWh74ZykdyrNYQ9n4BywcJ?usp=sharing)**, then register
them with Ollama as shown in [`finetune/README.md`](finetune/README.md) (SFT) and
[`finetune/dpo/README.md`](finetune/dpo/README.md) (DPO):

| Model | File | Ollama name |
|---|---|---|
| SFT (QLoRA) | `qwen2.5-3b-haggle-q4_k_m.gguf` | `qwen2.5-3b-haggle` |
| DPO v2 | `qwen2.5-3b-haggle-dpo-q4_k_m.gguf` | `qwen2.5-3b-haggle-dpo` |
| DPO v3 | `qwen2.5-3b-haggle-dpo-v3-q4_k_m.gguf` | `qwen2.5-3b-haggle-dpo-v3` |
| DPO v4 | `qwen2.5-3b-haggle-dpo-v4-q4_k_m.gguf` | `qwen2.5-3b-haggle-dpo-v4` |

For the SFT model, the short version is:

```bash
mkdir -p runs/finetune && mv qwen2.5-3b-haggle-q4_k_m.gguf runs/finetune/
cp finetune/Modelfile runs/finetune/Modelfile
(cd runs/finetune && ollama create qwen2.5-3b-haggle -f Modelfile)
```

## 2. Variables and configuration

### Environment variables (`.env`)

`.env` is git-ignored and loaded automatically (`haggle/config.py`). Each model entry in a config names the variable
that holds its key (`api_key_env`).

| Variable | Needed for | Value |
|---|---|---|
| `OPENROUTER_API_KEY` | Every hosted model: the main seller, the buyer pool and the judge | Your OpenRouter key |
| `OLLAMA_API_KEY` | Every local model | Any non-empty string (default `ollama`); Ollama ignores it, but the client needs one |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | Nothing by default | Only if you add a model entry with `api_key_env` pointing at them |

Experiments that use only local models and scripted or rule-based buyers (`sanity`, every `attacks_*` run on a local
seller, `closing_check`, the demo) need no API key.

### Models (`configs/models.yaml`)

Every experiment reads its models from this one file, so **no model names, prices or paths are hard-coded**. The
models used in the report:

| Key | Role | Model | Where |
|---|---|---|---|
| `main_seller` | Hosted seller for RQ1, RQ3, RQ4, strong side of RQ5 | `anthropic/claude-haiku-5.5` | OpenRouter |
| `buyer_pool` | LLM buyers (fixed across all conditions) | `openai/gpt-6-luna` | OpenRouter |
| `judge` | Honesty judge (different family from the sellers) | `xiaomi/mimo-v2.6-flash` | OpenRouter |
| `weak` | Untuned base model, weak side of RQ5 | `qwen2.5:3b` | Ollama |
| `finetuned` | SFT model, Agents C and D | `qwen2.5-3b-haggle` | Ollama |
| `finetuned_dpo`, `finetuned_dpo_v3`, `finetuned_dpo_v4` | DPO versions | `qwen2.5-3b-haggle-dpo[-v3/-v4]` | Ollama |

Each entry has these fields:

| Field | Meaning |
|---|---|
| `provider` | `openai_compatible` (OpenRouter, Ollama, or any other OpenAI-compatible endpoint) |
| `model` | The provider's model id |
| `base_url` | `https://openrouter.ai/api/v1` or `http://localhost:11434/v1` for Ollama |
| `api_key_env` | Name of the environment variable holding the key |
| `temperature`, `max_tokens` | Sampling settings. Always set `max_tokens`: constrained decoding on small models can loop |
| `price_in_per_mtok`, `price_out_per_mtok` | USD per million tokens, used for cost logging and the budget cap |

The runner refuses to start on a `SET_ME` placeholder, or on a hosted model priced at $0, because the budget cap
works from these prices. To swap in a different model, change the entry here; every config picks it up.

### Experiment configs (`configs/<experiment>.yaml`)

One file per experiment. The main fields:

| Field | Meaning |
|---|---|
| `experiment_name` | Output goes to `runs/<experiment_name>/<timestamp>/` |
| `seed` | Seeds scenario sampling, persona assignment and the LLM `seed` parameter |
| `max_budget_usd` | Safety cap: the runner aborts when estimated spend passes it. `0.0` for all-local runs |
| `models_file`, `listings_path`, `log_dir` | Inputs and output folder |
| `engine.max_turns_per_side` | Turn limit (10 in experiments) |
| `engine.max_parse_retries` | Retries on a malformed or illegal move before it counts as a format failure |
| `guardrails.floor_protection` | Engine floor guardrail: blocks any seller accept or offer below the floor. Off by default so the baselines show the failure |
| `matrix.listings`, `seeds`, `listing_sample` | Which scenarios are played |
| `matrix.sellers`, `matrix.buyers` | Agents with their switches, e.g. `{agent: agentic_seller, model: main_seller, use_tools: false, guard: {leak_filter: true, grounding: block}}`; `{agent: attacks, twins: true}` expands to every attack script and its twin |
| `summary_group_by`, `bootstrap` | How `summary.txt` groups results, and the bootstrap settings for the 95% CIs |

## 3. Quick start: tests and demo

```bash
uv run pytest                          # engine, metrics, agents, guardrails and fine-tuning helpers; no model calls
uv run streamlit run app/demo.py       # demo UI at http://localhost:8501 (needs Ollama running)
```

`pytest` includes the sanity-agent tests: rule-based agents play full games and the metrics must return their known
values (e.g. surplus share 1.0 and 0.0) before any LLM result is trusted. Agent tests use a fake LLM client, so they
need neither Ollama nor a key.

The demo uses `configs/demo.yaml` (local models only, so it costs nothing). Its sidebar picks the listing, seed,
seller model and agent, attack presets and the floor guardrail, and a "Behind the scenes" panel shows the floor,
the buyer's budget, tool calls and the agent's plan. The **Results** page browses any run under `runs/`. Each demo
session writes its games to `runs/demo/<timestamp>/`.

## 4. Running the experiments

Every experiment is one command:

```bash
uv run python -m haggle.runner configs/<experiment>.yaml [--dry-run] [--resume runs/<experiment>/<timestamp>]
```

- `--dry-run` prints the number of games without playing anything.
- `--resume` finishes an interrupted run, keeping its finished games. It refuses a run of a different config.

Run them in this order. "Key" means `OPENROUTER_API_KEY`; the fine-tuned models must be registered first (section 1).

| Step | Config | What | Games | Needs |
|---|---|---|---|---|
| 1 | `sanity` | Rule-based agents with known outcomes. **Must pass first** | 240 | nothing |
| 2 | `pilot` | Agent A vs fair/tough/lowballer: measures the real cost per game | 51 | key |
| 3 | `self_play` | Calibration: same model both sides, seller surplus should be near 0.5 | 60 | key |
| 4 | `replay` | Recorded pilot buyer turns replayed against sellers A and B | 60 | key |
| 5 | `tournament` | **RQ1** ablation: A, A+tools, A+plan, B × 3 personas × 20 scenarios | 240 | key |
| 6 | `honesty` | **RQ4**: honesty instruction off, paired with `tournament` | 120 | key |
| 7 | `attacks` | **RQ3**: 50 attack scripts + 50 plain twins × A/B × defences off/on | 400 | key |
| 8 | `asymmetry` | **RQ5**: strong vs weak model, same design, both directions | 120 | key, `qwen2.5:3b` |
| 9 | `opponent` | **RQ5**: the agentic buyer as a stronger opponent | 60 | key |
| 10 | `extension_solo` | **RQ6** baseline: 40 markets, one buyer each | 40 | key |
| 11 | `extension` | **RQ6** competition: 40 markets, three buyers each | 120 | key |
| 12 | `tournament_base` | RQ1/RQ2 on the untuned `qwen2.5:3b` (only buyers cost money) | 240 | key, `qwen2.5:3b` |
| 13 | `attacks_base` | RQ3 on the untuned model | 400 | `qwen2.5:3b` |
| 14 | `tournament_finetuned` | **RQ2**: Agents C and D (SFT model) on the same scenarios | 120 | key, SFT model |
| 15 | `tournament_finetuned_ablation` | RQ1 on the SFT model: A+tools, A+plan | 120 | key, SFT model |
| 16 | `attacks_finetuned` | RQ2 + RQ3 on the SFT model | 400 | SFT model |
| 17 | `attacks_floorguard` | Engine floor guardrail **on**, Agent A on the base and SFT models | 200 | base + SFT models |
| 18 | `tournament_dpo`, `attacks_dpo` | DPO v2 | 240, 400 | key (tournament), DPO v2 |
| 19 | `closing_check` | Free check that a local seller still closes deals (rule-based buyers) | 120 | local models |
| 20 | `attacks_dpo_v3` | DPO v3 | 400 | DPO v3 |

Runs on a local seller against scripted attackers cost nothing; with the hosted models above, the whole set fits in
about $6. Budgets in the configs are safety caps, so review them against the pilot's cost per game.

**Output.** Each run writes `runs/<experiment>/<timestamp>/` with the resolved `config.yaml`, `games.jsonl` (one
`GameResult` per game), `llm_calls.jsonl` (every model call with tokens, latency and cost), `summary.json` /
`summary.txt` (mean and bootstrap 95% CI per metric) and `manifest.json` (spend, cost per game, crashed games, budget
abort). Crashed games are left out of the statistics. The committed runs leave out `llm_calls.jsonl`.

**Attack effect** is success under the attack minus success under its plain twin (same prices, neutral words), paired
on script and scenario (`metrics.attack_effect`). The twins matter: small models sometimes "lose" to the plain twin
too.

## 5. Judging and analysis

**Honesty judge (RQ4).** Run the judge on each run that needs honesty verdicts, then check it against 30
hand-labelled games (`labels/hand_labels_30.csv`; rules in `haggle/prompts/judge_honesty.md`):

```bash
uv run python -m haggle.judge judge configs/judge.yaml runs/tournament/<timestamp>     # writes honesty.jsonl in that run
# ...repeat for honesty, tournament_finetuned, attacks, attacks_finetuned, pilot_sol, tournament_base,
#    tournament_finetuned_ablation, tournament_dpo, attacks_dpo
cat runs/*/*/honesty.jsonl > labels/all_verdicts.jsonl
uv run python -m haggle.judge agree labels/hand_labels_30.csv labels/all_verdicts.jsonl
uv run python -m haggle.judge export runs/<experiment>/<timestamp> --out labels/to_label.csv --n 100   # sample games for hand-labelling
```

**Report tables and charts.** `uv sync --group analysis`, then open `analysis/results.ipynb` with the project's
`.venv` kernel. It reads the frozen runs listed in its `RUNS` cell (the committed runs; change a path there to use a
new run) through `analysis/report.py`, and saves the figures to `analysis/figures/`. The demo's Results page
shows the same summaries interactively.

## 6. Fine-tuning and DPO

RQ2 fine-tunes **Qwen2.5-3B-Instruct**:

- **SFT** with QLoRA on seller turns from CraigslistBargain, replayed through our engine so training prompts match
  live ones. Instructions and both training routes (a Colab/CUDA notebook or MLX):
  [`finetune/README.md`](finetune/README.md). Results: [`finetune/SFT_RESULTS.md`](finetune/SFT_RESULTS.md).
- **DPO** on top of the SFT model, three versions, each in its own folder with its pairs (`train.jsonl`,
  `valid.jsonl`) and training notebook: [`finetune/dpo/`](finetune/dpo). Instructions and results:
  [`finetune/dpo/README.md`](finetune/dpo/README.md). The full design log is `finetune/DPO_V3.md`.

| Step | Command |
|---|---|
| Build the SFT data | `uv run python -m finetune.prepare_data` |
| Train SFT | `finetune/qlora_colab.ipynb` (any CUDA GPU) or `bash finetune/run_train.sh` (MLX) |
| Compare tuned vs stock | `uv run python -m finetune.compare_models qwen2.5:3b qwen2.5-3b-haggle` |
| Build DPO pairs | `uv run python -m finetune.build_dpo_pairs_v4 --seed 42` (v4; see the DPO README for v2/v3) |
| Audit DPO pairs | `uv run python -m finetune.audit_dpo_pairs finetune/dpo/v4/train.jsonl` |
| Train DPO | `finetune/dpo/<version>/dpo_colab.ipynb` |
| Check DPO decisions ($0) | `uv run python -m finetune.check_decisions finetune/dpo/v4/valid.jsonl --models finetuned finetuned_dpo_v4` |

The SFT notebook reads optional `HAGGLE_*` environment variables (data and output folders, batch size, smoke-test
steps), listed in `finetune/README.md`.

## 7. Reproducibility

- **Seeds** come from each config (`seed`, `matrix.seeds`, `bootstrap.seed`); the DPO pair builders take `--seed 42`
  and rebuild the committed pairs byte for byte.
- **Model versions** are pinned: hosted models list the version OpenRouter served on 7 Nov 2026 in
  `configs/models.yaml`; Qwen2.5-3B-Instruct is pinned to revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`; the MLX
  base weights to fixed revisions in `finetune/download_base.py`; every fine-tuned GGUF has its SHA-256 in the
  READMEs. `uv.lock` pins the Python dependencies.
- **Not bit-identical:** not every hosted model honours the `seed` parameter, so replays against hosted models can
  differ slightly.
- **Prompts** are versioned files in `haggle/prompts/*.md`, and every run keeps a copy of its resolved config.

## 8. Repository layout

| Path | What |
|---|---|
| `haggle/schemas.py` | Pydantic models: `Listing`, `Scenario`, `Action`, `Turn`, `GameResult` |
| `haggle/engine.py` | Turn loop and rules: standing offers, accept/offer/reject/walk away, retries, guardrails, simulated time |
| `haggle/orchestrator.py` | One seller against several buyers (RQ6) |
| `haggle/llm.py` | OpenAI-compatible client (Ollama, OpenRouter) with JSON-schema output, retries, usage logging and budget cap |
| `haggle/config.py` | Loads configs and `.env`, builds model clients |
| `haggle/agents/` | `prompted_seller.py` (Agent A), `agentic.py` (Agent B and the agentic buyer), `buyers.py` (LLM personas), `attackers.py` (red-team buyers and twins), `sanity.py` (rule-based test agents) |
| `haggle/guardrails.py` | Agent-side guardrails: injection wrapping, grounding check, floor-leak filter (`GuardedAgent`) |
| `haggle/tools/` | `price_lookup` (BM25 over comparables), calculator, walk-away estimator |
| `haggle/prompts/` | Prompt templates (`string.Template`) |
| `haggle/scenarios.py` | Seeded floor/budget generation and scenario batches |
| `haggle/metrics.py` | Deal rate, surplus share, seller payoff, below-floor deals, floor leaks, attack effect, bootstrap CIs |
| `haggle/judge.py` | LLM honesty judge, hand-label export and judge–human agreement |
| `haggle/runner.py` | Runs an experiment matrix from a config |
| `app/` | Streamlit demo (`demo.py`) and Results page |
| `analysis/` | `results.ipynb`, `report.py` and the report figures |
| `configs/` | `models.yaml` plus one config per experiment |
| `data/listings_sg.json` | 104 synthetic listings, each with at least one flaw |
| `data/attacks.json` | 50 attack scripts in 7 categories; each plain twin is generated in code |
| `data/comparables.json` | 416 price comparables for `price_lookup` (sources in `data/README.md`) |
| `finetune/` | SFT pipeline (see `finetune/README.md`) and DPO builders, audit and checks |
| `finetune/dpo/v2`, `v3`, `v4` | DPO pairs and training notebook for each version |
| `labels/` | Hand labels for validating the judge |
| `runs/` | The runs used in the report |
| `tests/` | `pytest` suite |

### Using the agents in code

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

Every agent has `name`, `model` and `act(observation) -> AgentReply`, so the engine, orchestrator and runner treat
them the same way. `client` comes from `haggle.config.make_client`.

## Data and licences

- Listings are synthetic, written for this project. Price comparables come from the non-marketplace sources listed
  in `data/README.md`. No marketplace was scraped.
- **CraigslistBargain** (He et al., 2018) is downloaded by `finetune/prepare_data.py` and not redistributed; its
  licence is listed as unknown on Hugging Face.
- **Qwen2.5-3B** is under the Qwen Research Licence, so the fine-tuned models are for research use only.
