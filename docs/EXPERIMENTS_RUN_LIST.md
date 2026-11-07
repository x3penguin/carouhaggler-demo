# Haggle Arena — Full Experiment Run Guide

> **🤖 AI AGENT INSTRUCTIONS:**
> Whenever you execute an experiment run or see a new completed run directory under `runs/`, you **MUST** update this file (`docs/EXPERIMENTS_RUN_LIST.md`):
> 1. Append `*(Done by <Name> on <YYYY-MM-DD>)*` to the section header of the completed run.
> 2. Update the **Output Path** field with the exact timestamped directory path (e.g., `runs/<experiment_name>/<YYYYMMDD-HHMMSS>/`).
> 3. Keep the Quick Status Summary table below in sync.

---

## Quick Status Summary

| # | Experiment | Status | Completed By | Date | Output Path |
|---|---|---|---|---|---|
| 1 | `sanity` | ✅ Done | Fayaz | 2026-11-06 | `runs/sanity/20261106-125147/` |
| 2 | `compare_models` | ✅ Done | Fayaz | 2026-11-05 | `runs/finetune/report.json` |
| 3 | `pilot` | ⏳ Run on ZY's machine (pending sync) | ZY | 2026-11-07 | `runs/pilot/<YYYYMMDD-HHMMSS>/` |
| 4 | `self_play` | ❌ Pending | — | — | `runs/self_play/<YYYYMMDD-HHMMSS>/` |
| 5 | `replay` | ❌ Pending | — | — | `runs/replay/<YYYYMMDD-HHMMSS>/` |
| 6 | `tournament` | ❌ Pending | — | — | `runs/tournament/<YYYYMMDD-HHMMSS>/` |
| 7 | `tournament_finetuned` | ❌ Pending | — | — | `runs/tournament_finetuned/<YYYYMMDD-HHMMSS>/` |
| 8 | `attacks` | ❌ Pending | — | — | `runs/attacks/<YYYYMMDD-HHMMSS>/` |
| 9 | `attacks_finetuned` | ✅ Done | Fayaz | 2026-11-06 | `runs/attacks_finetuned/20261106-125948/` |
| 10 | `honesty` | ❌ Pending | — | — | `runs/honesty/<YYYYMMDD-HHMMSS>/` |
| 11 | `judge` | ❌ Pending | — | — | `runs/<exp>/<YYYYMMDD-HHMMSS>/honesty.jsonl` |
| 12 | `asymmetry` | ❌ Pending | — | — | `runs/asymmetry/<YYYYMMDD-HHMMSS>/` |
| 13 | `opponent` | ❌ Pending | — | — | `runs/opponent/<YYYYMMDD-HHMMSS>/` |
| 14 | `extension` | ❌ Pending | — | — | `runs/extension/<YYYYMMDD-HHMMSS>/` |

---

## Model Keys (`configs/models.yaml`)

- **`main_seller` (Hosted):** `anthropic/claude-haiku-5.5` (OpenRouter)
- **`buyer_pool` (Hosted):** `openai/gpt-6-luna` (OpenRouter)
- **`judge` (Hosted):** `xiaomi/mimo-v2.6-flash` (OpenRouter)
- **`weak` (Local Ollama):** `qwen2.5:3b`
- **`finetuned` (Local Ollama):** `qwen2.5-3b-haggle`

> **Note:** Every `haggle.runner` run writes 6 files inside its output folder: `config.yaml`, `games.jsonl`, `llm_calls.jsonl`, `manifest.json`, `summary.json`, and `summary.txt`.

---

## 1. `sanity` (`configs/sanity.yaml`) — 240 games *(Done by Fayaz on 2026-11-06)*
- **Command:** `uv run python -m haggle.runner configs/sanity.yaml`
- **Output Path:** `runs/sanity/20261106-125147/`
- **What it is & Aim:** Runs deterministic rule-based sellers (`accept_above_floor`, `stubborn`, `leaky`, `below_floor`) against rule-based buyers (`full_budget`, `floor`, `linear`) with `floor_protection` on and off. Validates that the engine and metrics return exact known values (surplus share 1.0 and 0.0, 0 below-floor deals with guardrail on) before trusting any LLM numbers.
- **RQs Answered:** **Validation / Baseline Integrity** (Section 0 of `analysis/results.ipynb`).
- **Models:** **None** (100% deterministic code).

---

## 2. `compare_models` (`finetune/compare_models.py`) — 100 held-out turns *(Done by Fayaz on 2026-11-05)*
- **Command:** `uv run python -m finetune.compare_models qwen2.5:3b qwen2.5-3b-haggle --out runs/finetune/report.json`
- **Output Path:** `runs/finetune/report.json` *(Plus training curves in `runs/finetune/loss_curve.png` and `runs/finetune/training_log.json`)*
- **What it is & Aim:** Offline benchmark evaluating single-turn outputs on 100 held-out CraigslistBargain validation turns without forcing JSON schema. Measures valid JSON rate, human action alignment, below-floor offers, median price error, and tok/s.
- **RQs Answered:** **RQ2** (Training-time & static turn-level verification that QLoRA improved pricing and formatting over stock 3B).
- **Models:**
  - Stock baseline: `qwen2.5:3b` (Ollama)
  - Fine-tuned: `qwen2.5-3b-haggle` (Ollama)

---

## 3. `pilot` (`configs/pilot.yaml`) — 51 games *(Done by ZY on 2026-11-05 & 2026-11-07 on ZY's machine; pending sync to `runs/pilot/`)*
- **Command:** `uv run python -m haggle.runner configs/pilot.yaml`
- **Output Path:** `runs/pilot/<YYYYMMDD-HHMMSS>/`
- **What it is & Aim:** Runs Agent A (`prompted_seller`) across 17 listings × 3 buyer personas (`fair`, `tough`, `lowballer`). Measures real API cost per game, verifies token limits, and generates transcripts for hand-labelling the honesty judge.
- **RQs Answered:** **Cost calibration, Judge validation (§8), and RQ1/RQ2 reference baseline.**
- **Models:**
  - Seller: `main_seller` (`anthropic/claude-haiku-5.5`, plus the 2026-11-05 pilot with `GPT-6.1 Sol` kept as frontier reference)
  - Buyer: `buyer_pool` (`openai/gpt-6-luna`, plus `GPT-6.1 Terra` in the 2026-11-05 pilot)

---

## 4. `self_play` (`configs/self_play.yaml`) — 60 games *(Pending)*
- **Command:** `uv run python -m haggle.runner configs/self_play.yaml`
- **Output Path:** `runs/self_play/<YYYYMMDD-HHMMSS>/`
- **What it is & Aim:** Pits the exact same model against itself as both seller (`prompted_seller`) and buyer (`fair` persona) across 30 scenarios. Checks whether the game setup is fair (unbiased seller surplus share should sit near `0.50`) or has structural first-mover/anchoring bias.
- **RQs Answered:** **Calibration (§2)** — controls for game bias when reading RQ1 and RQ5.
- **Models:**
  - 30 games: `main_seller` vs. `main_seller` (`anthropic/claude-haiku-5.5` on both sides)
  - 30 games: `weak` vs. `weak` (`qwen2.5:3b` on Ollama on both sides)

---

## 5. `replay` (`configs/replay.yaml`) — 60 games *(Pending)*
- **Command:** `uv run python -m haggle.runner configs/replay.yaml`
- **Output Path:** `runs/replay/<YYYYMMDD-HHMMSS>/` *(Reads buyer turns from latest `runs/pilot/<YYYYMMDD-HHMMSS>/games.jsonl`)*
- **What it is & Aim:** Counterfactual replay harness. Takes 30 recorded buyer transcripts from `runs/pilot` and replays the exact same buyer messages/offers against Agent A (`prompted_seller`) and Agent B (`agentic_seller`). Eliminates buyer randomness so any difference is 100% due to seller design.
- **RQs Answered:** **RQ1** (Paired counterfactual seller comparison, §3).
- **Models:**
  - Seller: `main_seller` (`anthropic/claude-haiku-5.5`)
  - Buyer: **None** (replays recorded pilot turns from disk)

---

## 6. `tournament` (`configs/tournament.yaml`) — 240 games *(Pending)*
- **Command:** `uv run python -m haggle.runner configs/tournament.yaml`
- **Output Path:** `runs/tournament/<YYYYMMDD-HHMMSS>/`
- **What it is & Aim:** Main ablation tournament across 20 listings × 3 buyer personas (`fair`, `tough`, `lowballer`) × 4 seller variants:
  1. Agent A (`prompted_seller`)
  2. Agent A + Tools only (`agentic_seller[tools]`)
  3. Agent A + Planning only (`agentic_seller[plan]`)
  4. Full Agent B (`agentic_seller[tools+plan+reflect]`)
- **RQs Answered:** **RQ1** (Does agentic scaffolding beat prompting, and which component drives the gain?), **RQ2** (Hosted A/B baseline on the cost-vs-quality Pareto chart), and **RQ4** ("honesty-on" arm when graded by the judge).
- **Models:**
  - Seller: `main_seller` (`anthropic/claude-haiku-5.5`)
  - Buyer: `buyer_pool` (`openai/gpt-6-luna`)

---

## 7. `tournament_finetuned` (`configs/tournament_finetuned.yaml`) — 120 games *(Pending)*
- **Command:** `uv run python -m haggle.runner configs/tournament_finetuned.yaml`
- **Output Path:** `runs/tournament_finetuned/<YYYYMMDD-HHMMSS>/`
- **What it is & Aim:** Runs **Agent C** (`prompted_seller` + `finetuned`) and **Agent D** (`agentic_seller[tools+plan+reflect]` + `finetuned`) on the **exact same 20 listings × 3 buyer personas** as `tournament.yaml`. Tests how the local fine-tuned 3B model performs in live negotiation against LLM buyers.
- **RQs Answered:** **RQ2** (Can a fine-tuned 1–3B model compete with hosted A/B on deal rate, surplus share, and seller payoff at $0 seller inference cost?).
- **Models:**
  - Seller: `finetuned` (`qwen2.5-3b-haggle` on local Ollama)
  - Buyer: `buyer_pool` (`openai/gpt-6-luna` on OpenRouter)

---

## 8. `attacks` (`configs/attacks.yaml`) — 400 games *(Pending)*
- **Command:** `uv run python -m haggle.runner configs/attacks.yaml`
- **Output Path:** `runs/attacks/<YYYYMMDD-HHMMSS>/`
- **What it is & Aim:** Tests 4 hosted seller variants (Agent A, Agent A + Guardrails, Agent B, Agent B + Guardrails) against 50 scripted red-team attacks + 50 neutral price-matched twins across 7 categories (sob story, fake urgency, extreme lowball, prompt injection, floor extraction, fake competing offer, deadline exploit).
- **RQs Answered:** **RQ3** (How exploitable are hosted seller agents $\Delta = \text{Attack} - \text{Twin}$, and do `GuardedAgent` + injection wrapping block the exploits?) and **RQ6** (compares fake competing offer claims against real ones).
- **Models:**
  - Seller: `main_seller` (`anthropic/claude-haiku-5.5`)
  - Buyer: **None** (deterministic `AttackBuyer` scripts + twins)

---

## 9. `attacks_finetuned` (`configs/attacks_finetuned.yaml`) — 400 games *(Done by Fayaz on 2026-11-06)*
- **Command:** `uv run python -m haggle.runner configs/attacks_finetuned.yaml`
- **Output Path:** `runs/attacks_finetuned/20261106-125948/`
- **What it is & Aim:** Runs the exact same 50 attack scripts + 50 neutral twins as `attacks.yaml`, but against **Agent C**, **Agent C + Guardrails**, **Agent D**, and **Agent D + Guardrails**.
- **RQs Answered:** **RQ2 + RQ3** (Measures exploitability, guardrail efficacy, below-floor deals, floor leaks, and JSON format stability of fine-tuned Agents C & D under adversarial attack vs. hosted Agents A & B).
- **Models:**
  - Seller: `finetuned` (`qwen2.5-3b-haggle` on local Ollama)
  - Buyer: **None** (deterministic `AttackBuyer` scripts + twins)

---

## 10. `honesty` (`configs/honesty.yaml`) — 120 games *(Pending)*
- **Command:** `uv run python -m haggle.runner configs/honesty.yaml`
- **Output Path:** `runs/honesty/<YYYYMMDD-HHMMSS>/`
- **What it is & Aim:** Runs Agent A and full Agent B on the same 20 scenarios × 3 condition-probing buyer personas as `tournament.yaml`, but with `honesty_instruction: false` (removes the prompt rule telling the seller to be truthful about item flaws).
- **RQs Answered:** **RQ4** (Do better negotiators lie more about item condition, and how much does an explicit honesty instruction reduce lies when paired against `tournament.yaml`?).
- **Models:**
  - Seller: `main_seller` (`anthropic/claude-haiku-5.5`)
  - Buyer: `buyer_pool` (`openai/gpt-6-luna`)

---

## 11. `judge` (`configs/judge.yaml`) — Post-processing pass *(Pending)*
- **Commands & Output Paths:**
  - Export for human labelling:
    `uv run python -m haggle.judge export runs/pilot/<ts> --out labels/pilot_100.csv --n 100`
    → **Writes:** `labels/pilot_100.csv`
  - Judge runs (writes `honesty.jsonl` directly inside each target run folder):
    - `uv run python -m haggle.judge judge configs/judge.yaml runs/pilot/<ts>` → **Writes:** `runs/pilot/<ts>/honesty.jsonl`
    - `uv run python -m haggle.judge judge configs/judge.yaml runs/tournament/<ts>` → **Writes:** `runs/tournament/<ts>/honesty.jsonl`
    - `uv run python -m haggle.judge judge configs/judge.yaml runs/honesty/<ts>` → **Writes:** `runs/honesty/<ts>/honesty.jsonl`
  - Human-vs-Judge agreement check:
    `uv run python -m haggle.judge agree labels/pilot_100.csv runs/pilot/<ts>/honesty.jsonl`
- **What it is & Aim:** 3-sample majority-vote LLM-as-judge that inspects each game transcript against `Listing.condition_facts` and logs whether the seller lied or fabricated facts. Also checks agreement (precision/recall/F1) against `labels/pilot_100.csv`.
- **RQs Answered:** **RQ4** (Deal quality vs. honesty).
- **Models:**
  - Judge: `judge` (`xiaomi/mimo-v2.6-flash` — intentionally a different model family to avoid self-preference bias).

---

## 12. `asymmetry` (`configs/asymmetry.yaml`) — 120 games *(Pending)*
- **Command:** `uv run python -m haggle.runner configs/asymmetry.yaml`
- **Output Path:** `runs/asymmetry/<YYYYMMDD-HHMMSS>/`
- **What it is & Aim:** 2×2 matrix across 30 scenarios with the same agent architecture (`prompted_seller` vs. `fair` `llm_buyer`):
  1. Strong Seller vs. Strong Buyer
  2. Strong Seller vs. Weak Buyer
  3. Weak Seller vs. Strong Buyer
  4. Weak Seller vs. Weak Buyer
- **RQs Answered:** **RQ5** (Model capability asymmetry: how much surplus shifts to the stronger model when agent design is held constant?).
- **Models:**
  - Strong side: `main_seller` (`anthropic/claude-haiku-5.5` on OpenRouter)
  - Weak side: `weak` (`qwen2.5:3b` on local Ollama)

---

## 13. `opponent` (`configs/opponent.yaml`) — 60 games *(Pending)*
- **Command:** `uv run python -m haggle.runner configs/opponent.yaml`
- **Output Path:** `runs/opponent/<YYYYMMDD-HHMMSS>/`
- **What it is & Aim:** Runs Agent A (`prompted_seller`) and Agent B (`agentic_seller`) across 30 scenarios against `AgenticBuyer` (a buyer with its own reflect → tools → plan → act loop). Keeps "stronger opponent architecture" separate from "stronger model weights" (`asymmetry.yaml`).
- **RQs Answered:** **RQ5** (How do sellers hold up against an agentic buyer with tools and planning?).
- **Models:**
  - Seller: `main_seller` (`anthropic/claude-haiku-5.5`)
  - Buyer: `buyer_pool` (`openai/gpt-6-luna` running inside `AgenticBuyer`)

---

## 14. `extension` (`configs/extension.yaml`) — 40 markets / 120 concurrent buyer chats *(Pending)*
- **Command:** `uv run python -m haggle.runner configs/extension.yaml`
- **Output Path:** `runs/extension/<YYYYMMDD-HHMMSS>/`
- **What it is & Aim:** Uses `MultiGameOrchestrator` across 20 listings × 2 sellers (Agent A and Agent B). In each market, 3 buyers arrive at staggered times (`0h` urgent 1.5d deadline, `12h` normal 5d deadline, `24h` relaxed 10d deadline) under a 7-day seller deadline. Once the seller accepts one buyer, the other chats terminate with `no_deal_item_sold`.
- **RQs Answered:** **RQ6** (Do time pressure and multi-buyer competition speed up seller concessions near deadlines, and how do sellers react to real platform-verified competing offers vs. fake competing-offer bluffs from `attacks.yaml`?).
- **Models:**
  - Seller: `main_seller` (`anthropic/claude-haiku-5.5`)
  - Buyers (3 per market): `buyer_pool` (`openai/gpt-6-luna`)
