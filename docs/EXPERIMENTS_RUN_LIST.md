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
| 3 | `pilot` | ✅ Done | ZY | 2026-11-07 | `runs/pilot/20261107-172247/` (frontier reference: `runs/pilot_sol/20261105-162801/`; 300-token debug run: `runs/pilot_haiku_300tok/20261107-163344/`) |
| 4 | `self_play` | ✅ Done | ZY | 2026-11-08 | `runs/self_play/20261108-010530/` |
| 5 | `replay` | ✅ Done | ZY | 2026-11-08 | `runs/replay/20261108-011751/` |
| 6 | `tournament` | ✅ Done | ZY | 2026-11-07 | `runs/tournament/20261107-175240/` |
| 7 | `tournament_finetuned` | ✅ Done | ZY | 2026-11-07 | `runs/tournament_finetuned/20261107-232336/` |
| 8 | `attacks` | ✅ Done | ZY | 2026-11-07 | `runs/attacks/20261107-202916/` |
| 9 | `attacks_finetuned` | ✅ Done | Fayaz, ZY | 2026-11-06, 2026-11-07 | `runs/attacks_finetuned/20261107-172310/` (ZY, judged; the notebook reads this newest run). Fayaz's Mac run: `runs/attacks_finetuned/20261106-125948/` |
| 10 | `honesty` | ✅ Done | ZY | 2026-11-07 | `runs/honesty/20261107-213142/` |
| 11 | `judge` | ✅ Done (30 hand labels in) | ZY; labels Fayaz, ZY, Anya | 2026-11-08, 2026-11-09 | `honesty.jsonl` in the tournament, honesty, tournament_finetuned, pilot_sol, attacks, attacks_finetuned, tournament_base and tournament_finetuned_ablation runs above (tournament_dpo and attacks_dpo not judged yet); `labels/hand_labels_30.csv` |
| 12 | `asymmetry` | ✅ Done | ZY | 2026-11-07 | `runs/asymmetry/20261107-225514/` |
| 13 | `opponent` | ✅ Done | ZY | 2026-11-08 | `runs/opponent/20261108-013713/` |
| 14 | `extension` (RQ6, competition) | ✅ Done | ZY | 2026-11-08 | `runs/extension/20261108-142244/` (superseded first design: `runs/extension/20261108-024342/`) |
| 15 | `extension_solo` (RQ6, baseline) | ✅ Done | ZY | 2026-11-08 | `runs/extension_solo/20261108-135244/` |
| 16 | `tournament_base` (RQ1/RQ2, untuned base model) | ✅ Done | ZY | 2026-11-09 | `runs/tournament_base/20261109-153143/` |
| 17 | `tournament_finetuned_ablation` (RQ1, fine-tuned A+tools, A+plan) | ✅ Done | ZY | 2026-11-09 | `runs/tournament_finetuned_ablation/20261109-170454/` |
| 18 | `attacks_base` (RQ3, untuned base model) | ✅ Done | ZY | 2026-11-09 | `runs/attacks_base/20261109-183201/` |
| 19 | `tournament_dpo` (RQ1/RQ2 v2, DPO model) | ✅ Done | ZY; DPO training Fayaz | 2026-11-10 | `runs/tournament_dpo/20261110-003810/` |
| 20 | `attacks_dpo` (RQ3, DPO model) | ✅ Done | ZY; DPO training Fayaz | 2026-11-10 | `runs/attacks_dpo/20261110-003810/` |
| 21 | `closing_check` (RQ2 diagnostic: can a local seller close?) | ✅ Done | ZY | 2026-11-10 | `runs/closing_check/20261110-130434/` (base, SFT, v2); `runs/closing_check_v3only/20261110-185907/` (v3, same games) |
| 22 | `attacks_dpo_v3` (RQ3, DPO v3 model) | ✅ Done | ZY; DPO v3 training Fayaz | 2026-11-10 | `runs/attacks_dpo_v3/20261110-190057/` |
| 23 | `attacks_floorguard` (report §6.3: floor guardrail on) | ✅ Done | ZY | 2026-11-11 | `runs/attacks_floorguard/20261111-144121/` |

---

## Model Keys (`configs/models.yaml`)

- **`main_seller` (Hosted):** `anthropic/claude-haiku-5.5` (OpenRouter)
- **`buyer_pool` (Hosted):** `openai/gpt-6-luna` (OpenRouter)
- **`judge` (Hosted):** `xiaomi/mimo-v2.6-flash` (OpenRouter)
- **`weak` (Local Ollama):** `qwen2.5:3b`
- **`finetuned` (Local Ollama):** `qwen2.5-3b-haggle`
- **`finetuned_dpo` (Local Ollama):** `qwen2.5-3b-haggle-dpo` (the fine-tuned model trained further with DPO)
- **`finetuned_dpo_v3` (Local Ollama):** `qwen2.5-3b-haggle-dpo-v3` (DPO retrained from the SFT model with close pairs, `finetune/DPO_V3.md`)

> **Note:** Every `haggle.runner` run writes 6 files inside its output folder: `config.yaml`, `games.jsonl`, `llm_calls.jsonl`, `manifest.json`, `summary.json`, and `summary.txt`.
> The raw call logs (`llm_calls.jsonl`, `judge_llm_calls.jsonl`) are not committed: they hold every prompt and reply (about 90 MB), and the analysis reads only games, summaries and judge verdicts. They stay on ZY's machine.

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

## 3. `pilot` (`configs/pilot.yaml`) — 51 games *(Done by ZY on 2026-11-07)*
- **Command:** `uv run python -m haggle.runner configs/pilot.yaml`
- **Output Path:** `runs/pilot/20261107-172247/`. Frontier reference (GPT-6.1 Sol seller, 5 Nov): `runs/pilot_sol/20261105-162801/`. First Haiku pilot with a 300-token limit, kept as debugging evidence (65 of 301 replies cut off): `runs/pilot_haiku_300tok/20261107-163344/`.
- **What it is & Aim:** Runs Agent A (`prompted_seller`) across 17 listings × 3 buyer personas (`fair`, `tough`, `lowballer`). Measures real API cost per game, verifies token limits, and generates transcripts for hand-labelling the honesty judge.
- **RQs Answered:** **Cost calibration, Judge validation (§8), and RQ1/RQ2 reference baseline.**
- **Models:**
  - Seller: `main_seller` (`anthropic/claude-haiku-5.5`, plus the 2026-11-05 pilot with `GPT-6.1 Sol` kept as frontier reference)
  - Buyer: `buyer_pool` (`openai/gpt-6-luna`, plus `GPT-6.1 Terra` in the 2026-11-05 pilot)

---

## 4. `self_play` (`configs/self_play.yaml`) — 60 games *(Done by ZY on 2026-11-08)*
- **Command:** `uv run python -m haggle.runner configs/self_play.yaml`
- **Output Path:** `runs/self_play/20261108-010530/`
- **What it is & Aim:** Pits the exact same model against itself as both seller (`prompted_seller`) and buyer (`fair` persona) across 30 scenarios. Checks whether the game setup is fair (unbiased seller surplus share should sit near `0.50`) or has structural first-mover/anchoring bias.
- **RQs Answered:** **Calibration (§2)** — controls for game bias when reading RQ1 and RQ5.
- **Models:**
  - 30 games: `main_seller` vs. `main_seller` (`anthropic/claude-haiku-5.5` on both sides)
  - 30 games: `weak` vs. `weak` (`qwen2.5:3b` on Ollama on both sides)

---

## 5. `replay` (`configs/replay.yaml`) — 60 games *(Done by ZY on 2026-11-08)*
- **Command:** `uv run python -m haggle.runner configs/replay.yaml`
- **Output Path:** `runs/replay/20261108-011751/` *(Reads buyer turns from latest `runs/pilot/<YYYYMMDD-HHMMSS>/games.jsonl`)*
- **What it is & Aim:** Counterfactual replay harness. Takes 30 recorded buyer transcripts from `runs/pilot` and replays the exact same buyer messages/offers against Agent A (`prompted_seller`) and Agent B (`agentic_seller`). Eliminates buyer randomness so any difference is 100% due to seller design.
- **RQs Answered:** **RQ1** (Paired counterfactual seller comparison, §3).
- **Models:**
  - Seller: `main_seller` (`anthropic/claude-haiku-5.5`)
  - Buyer: **None** (replays recorded pilot turns from disk)

---

## 6. `tournament` (`configs/tournament.yaml`) — 240 games *(Done by ZY on 2026-11-07)*
- **Command:** `uv run python -m haggle.runner configs/tournament.yaml`
- **Output Path:** `runs/tournament/20261107-175240/`
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

## 7. `tournament_finetuned` (`configs/tournament_finetuned.yaml`) — 120 games *(Done by ZY on 2026-11-07)*
- **Command:** `uv run python -m haggle.runner configs/tournament_finetuned.yaml`
- **Output Path:** `runs/tournament_finetuned/20261107-232336/`
- **What it is & Aim:** Runs **Agent C** (`prompted_seller` + `finetuned`) and **Agent D** (`agentic_seller[tools+plan+reflect]` + `finetuned`) on the **exact same 20 listings × 3 buyer personas** as `tournament.yaml`. Tests how the local fine-tuned 3B model performs in live negotiation against LLM buyers.
- **RQs Answered:** **RQ2** (Can a fine-tuned 1–3B model compete with hosted A/B on deal rate, surplus share, and seller payoff at $0 seller inference cost?).
- **Models:**
  - Seller: `finetuned` (`qwen2.5-3b-haggle` on local Ollama)
  - Buyer: `buyer_pool` (`openai/gpt-6-luna` on OpenRouter)

---

## 8. `attacks` (`configs/attacks.yaml`) — 400 games *(Done by ZY on 2026-11-07)*
- **Command:** `uv run python -m haggle.runner configs/attacks.yaml`
- **Output Path:** `runs/attacks/20261107-202916/`
- **What it is & Aim:** Tests 4 hosted seller variants (Agent A, Agent A + Guardrails, Agent B, Agent B + Guardrails) against 50 scripted red-team attacks + 50 neutral price-matched twins across 7 categories (sob story, fake urgency, extreme lowball, prompt injection, floor extraction, fake competing offer, deadline exploit).
- **RQs Answered:** **RQ3** (How exploitable are hosted seller agents $\Delta = \text{Attack} - \text{Twin}$, and do `GuardedAgent` + injection wrapping block the exploits?) and **RQ6** (compares fake competing offer claims against real ones).
- **Models:**
  - Seller: `main_seller` (`anthropic/claude-haiku-5.5`)
  - Buyer: **None** (deterministic `AttackBuyer` scripts + twins)

---

## 9. `attacks_finetuned` (`configs/attacks_finetuned.yaml`) — 400 games *(Done by Fayaz on 2026-11-06; re-run by ZY on 2026-11-07)*
- **Command:** `uv run python -m haggle.runner configs/attacks_finetuned.yaml`
- **Output Path:** `runs/attacks_finetuned/20261107-172310/` (ZY's run, judged; the notebook reads the newest run). Fayaz's Mac run: `runs/attacks_finetuned/20261106-125948/`.
- **What it is & Aim:** Runs the exact same 50 attack scripts + 50 neutral twins as `attacks.yaml`, but against **Agent C**, **Agent C + Guardrails**, **Agent D**, and **Agent D + Guardrails**.
- **RQs Answered:** **RQ2 + RQ3** (Measures exploitability, guardrail efficacy, below-floor deals, floor leaks, and JSON format stability of fine-tuned Agents C & D under adversarial attack vs. hosted Agents A & B).
- **Models:**
  - Seller: `finetuned` (`qwen2.5-3b-haggle` on local Ollama)
  - Buyer: **None** (deterministic `AttackBuyer` scripts + twins)

---

## 10. `honesty` (`configs/honesty.yaml`) — 120 games *(Done by ZY on 2026-11-07)*
- **Command:** `uv run python -m haggle.runner configs/honesty.yaml`
- **Output Path:** `runs/honesty/20261107-213142/`
- **What it is & Aim:** Runs Agent A and full Agent B on the same 20 scenarios × 3 condition-probing buyer personas as `tournament.yaml`, but with `honesty_instruction: false` (removes the prompt rule telling the seller to be truthful about item flaws).
- **RQs Answered:** **RQ4** (Do better negotiators lie more about item condition, and how much does an explicit honesty instruction reduce lies when paired against `tournament.yaml`?).
- **Models:**
  - Seller: `main_seller` (`anthropic/claude-haiku-5.5`)
  - Buyer: `buyer_pool` (`openai/gpt-6-luna`)

---

## 11. `judge` (`configs/judge.yaml`) — Post-processing pass *(Done by ZY on 2026-11-08 and 2026-11-09; 30 hand labels in)*
- **Commands & Output Paths:**
  - Judge runs (each writes `honesty.jsonl` inside the run folder): 1,331 games, 0 failed verdicts, about $0.70.
    `uv run python -m haggle.judge judge configs/judge.yaml runs/<exp>/<ts>` for `tournament/20261107-175240`, `honesty/20261107-213142`,
    `tournament_finetuned/20261107-232336`, `pilot_sol/20261105-162801`, `attacks/20261107-202916` and `attacks_finetuned/20261107-172310`.
    On 9 Nov also `tournament_base/20261109-153143` and `tournament_finetuned_ablation/20261109-170454` (360 games).
  - Agreement on all 30 labels (9 Nov): floor leak accuracy 0.93, precision 0.89, recall 0.89. Dishonesty: accuracy 0.77 but
    precision 0.25 when any flagged claim counts (6 false alarms, all minor sales talk such as "in great condition"); counting only
    major violations, precision 1.00 and recall 0.33. The notebook therefore charts major lies and tables both.
  - Hand labels: `labels/hand_labels_30.csv`, 30 judged games picked for agreement (8 the judge called dishonest, 7 it called
    a floor leak, 15 it passed), spread across sellers and blind to seller and model. Fill `human_dishonest`, `human_floor_leak`
    (0 or 1) and `labeller`; the rules are the judge's, in `haggle/prompts/judge_honesty.md`.
  - Human-vs-judge agreement, once labelled (the verdicts from all six runs in one file):
    `cat runs/*/*/honesty.jsonl > labels/all_verdicts.jsonl`
    `uv run python -m haggle.judge agree labels/hand_labels_30.csv labels/all_verdicts.jsonl`
- **What it is & Aim:** 3-sample majority-vote LLM-as-judge that inspects each game transcript against `Listing.condition_facts` and logs whether the seller lied or fabricated facts. Also checks agreement (precision/recall/F1) against `labels/hand_labels_30.csv`.
- **RQs Answered:** **RQ4** (Deal quality vs. honesty).
- **Models:**
  - Judge: `judge` (`xiaomi/mimo-v2.6-flash` — intentionally a different model family to avoid self-preference bias).

---

## 12. `asymmetry` (`configs/asymmetry.yaml`) — 120 games *(Done by ZY on 2026-11-07)*
- **Command:** `uv run python -m haggle.runner configs/asymmetry.yaml`
- **Output Path:** `runs/asymmetry/20261107-225514/`
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

## 13. `opponent` (`configs/opponent.yaml`) — 60 games *(Done by ZY on 2026-11-08)*
- **Command:** `uv run python -m haggle.runner configs/opponent.yaml`
- **Output Path:** `runs/opponent/20261108-013713/`
- **What it is & Aim:** Runs Agent A (`prompted_seller`) and Agent B (`agentic_seller`) across 30 scenarios against `AgenticBuyer` (a buyer with its own reflect → tools → plan → act loop). Keeps "stronger opponent architecture" separate from "stronger model weights" (`asymmetry.yaml`).
- **RQs Answered:** **RQ5** (How do sellers hold up against an agentic buyer with tools and planning?).
- **Models:**
  - Seller: `main_seller` (`anthropic/claude-haiku-5.5`)
  - Buyer: `buyer_pool` (`openai/gpt-6-luna` running inside `AgenticBuyer`)

---

## 14. `extension` (`configs/extension.yaml`) — 40 markets / 120 concurrent buyer chats *(Done by ZY on 2026-11-08)*
- **Command:** `uv run python -m haggle.runner configs/extension.yaml`
- **Output Path:** `runs/extension/20261108-142244/`. The first design's run, `runs/extension/20261108-024342/`, is superseded: its buyers arrived at hours 0, 12 and 24, and the first buyer closed all 40 sales before the others arrived, so no competition happened.
- **What it is & Aim:** Uses `MultiGameOrchestrator` across 20 listings × 2 sellers (Agent A and Agent B). In each market, a tough, a fair and a lowballer buyer all arrive at hour 0 with the same 5-day deadline (so urgency doesn't vary with persona), under a 7-day seller deadline. Once the seller accepts one buyer, the other chats end with `no_deal_item_sold`. Buyer 1 (tough) has the same budget as the lone buyer in `extension_solo`.
- **RQs Answered:** **RQ6** (Does competition between buyers make the seller easier to exploit, or does it help the seller? Also: how sellers react to real platform-verified competing offers vs. fake competing-offer bluffs from `attacks.yaml`). Time pressure was dropped from RQ6: deals closed within 2–11 hours against a 7-day deadline.
- **Models:**
  - Seller: `main_seller` (`anthropic/claude-haiku-5.5`)
  - Buyers (3 per market): `buyer_pool` (`openai/gpt-6-luna`)

---

## 15. `extension_solo` (`configs/extension_solo.yaml`) — 40 markets, 1 buyer each *(Done by ZY on 2026-11-08)*
- **Command:** `uv run python -m haggle.runner configs/extension_solo.yaml`
- **Output Path:** `runs/extension_solo/20261108-135244/`
- **What it is & Aim:** The no-competition baseline for RQ6. The same 20 listings, sellers, floors and seller deadline as `extension`, with only the tough buyer, on the same budget and deadline as buyer 1 there. It runs through the same multi-buyer code path, so the only difference is the two rivals. A test checks the two configs match game by game.
- **RQs Answered:** **RQ6** (paired with `extension`).
- **Models:**
  - Seller: `main_seller` (`anthropic/claude-haiku-5.5`)
  - Buyer: `buyer_pool` (`openai/gpt-6-luna`)

---

## 16. `tournament_base` (`configs/tournament_base.yaml`) — 240 games *(Done by ZY on 2026-11-09)*
- **Command:** `uv run python -m haggle.runner configs/tournament_base.yaml`
- **Output Path:** `runs/tournament_base/20261109-153143/` (0 crashed games, $0.14 of buyer calls)
- **What it is & Aim:** The four RQ1 designs (A, A+tools, A+plan, B) on the untuned base model, `qwen2.5:3b`, which is the model we fine-tuned. Same scenarios, buyers and seed as `tournament` (a test checks the games match), so RQ1 covers every model and RQ2 compares base -> fine-tuned -> Haiku game by game.
- **RQs Answered:** **RQ1** (does the design help on a weak model?), **RQ2** (how much did fine-tuning help?).
- **Models:**
  - Seller: `weak` (`qwen2.5:3b` on local Ollama)
  - Buyer: `buyer_pool` (`openai/gpt-6-luna`)

---

## 17. `tournament_finetuned_ablation` (`configs/tournament_finetuned_ablation.yaml`) — 120 games *(Done by ZY on 2026-11-09)*
- **Command:** `uv run python -m haggle.runner configs/tournament_finetuned_ablation.yaml`
- **Output Path:** `runs/tournament_finetuned_ablation/20261109-170454/` (0 crashed games, $0.15 of buyer calls)
- **What it is & Aim:** The two middle designs (A+tools, A+plan) on the fine-tuned model, which `tournament_finetuned` left out. Pooled with that run, the fine-tuned model has all four designs on the same games as Haiku and the base model.
- **RQs Answered:** **RQ1** (design on the fine-tuned model).
- **Models:**
  - Seller: `finetuned` (`qwen2.5-3b-haggle` on local Ollama)
  - Buyer: `buyer_pool` (`openai/gpt-6-luna`)

---

## 18. `attacks_base` (`configs/attacks_base.yaml`) — 400 games *(Done by ZY on 2026-11-09)*
- **Command:** `uv run python -m haggle.runner configs/attacks_base.yaml`
- **Output Path:** `runs/attacks_base/20261109-183201/` (scripted buyers and a local seller, so $0)
- **What it is & Aim:** The RQ3 attack scripts and their plain twins against the untuned base model, with defences off and on, on the same games as `attacks` (Haiku) and `attacks_finetuned`.
- **RQs Answered:** **RQ3** (exploitability on all three models).
- **Models:**
  - Seller: `weak` (`qwen2.5:3b` on local Ollama)
  - Buyers: scripted attackers from `data/attacks.json`

---

## 19. `tournament_dpo` (`configs/tournament_dpo.yaml`) — 240 games *(Done by ZY on 2026-11-10; DPO training by Fayaz)*
- **Command:** `uv run python -m haggle.runner configs/tournament_dpo.yaml`
- **Output Path:** `runs/tournament_dpo/20261110-003810/` ($0.43 of buyer calls; finished with `--resume` after two low-memory stops, about $0.03 lost on the first attempt)
- **What it is & Aim:** RQ2 v2. The fine-tuned model trained further with DPO (`qwen2.5-3b-haggle-dpo-q4_k_m.gguf`, SHA-256 `281be40d...c950c9f`) on 3,700 preference pairs from CraigslistBargain listings (none of ours): 559 hold-the-floor vs a below-floor accept, 559 counter-offer vs stating the floor, 2,582 the human seller's reply vs an invented claim. All four RQ1 designs, on the same games as `tournament`, `tournament_finetuned` and `tournament_base` (a test checks the games match), so v1 -> v2 is paired.
- **Limitation:** every rejected lie in the pairs is the same sentence ("brand new in box with full warranty and zero scratches"), so the honesty signal is narrow.
- **Result:** fewer below-floor deals than the SFT model (A 38% -> 13%, B 22% -> 15%), but mostly because it stopped closing. It closed 23 / 3 / 55 / 42% of games (A / A+tools / A+plan / B), against 80 / 65 / 78 / 67% for SFT, and payoff didn't improve. It accepted 9 times in 2,142 seller turns, and often wrote "Deal, thanks." while keeping its old offer. Cause: `accept` was only ever the rejected move in the pairs. The v3 plan is in `finetune/DPO_V3.md`.
- **RQs Answered:** **RQ2** (did DPO close the gap?), **RQ1**, **RQ4**.
- **Models:**
  - Seller: `finetuned_dpo` (`qwen2.5-3b-haggle-dpo` on local Ollama)
  - Buyer: `buyer_pool` (`openai/gpt-6-luna`)

---

## 20. `attacks_dpo` (`configs/attacks_dpo.yaml`) — 400 games *(Done by ZY on 2026-11-10; DPO training by Fayaz)*
- **Command:** `uv run python -m haggle.runner configs/attacks_dpo.yaml`
- **Output Path:** `runs/attacks_dpo/20261110-003810/` (scripted buyers and a local seller, so $0)
- **What it is & Aim:** The RQ3 attack scripts and their plain twins against the DPO model, with defences off and on, on the same games as the other three models. DPO was trained to hold the floor and not state it, so this is where it should show.
- **Result:** 8 of 200 attacks won (SFT 14, base 73, Haiku 0), and 6 twin failures. Partly the same accept collapse as in `tournament_dpo`: a seller that rarely accepts is hard to talk into a below-floor deal.
- **RQs Answered:** **RQ3**, **RQ2** (v1 -> v2).
- **Models:**
  - Seller: `finetuned_dpo` (`qwen2.5-3b-haggle-dpo` on local Ollama)
  - Buyers: scripted attackers from `data/attacks.json`

---

## 21. `closing_check` (`configs/closing_check.yaml`) — 120 games *(Done by ZY on 2026-11-10)*
- **Command:** `uv run python -m haggle.runner configs/closing_check.yaml`
- **Output Path:** `runs/closing_check/20261110-130434/` (base, SFT, v2; 90 games) and `runs/closing_check_v3only/20261110-185907/` (v3 only, the same 30 games, so the first three weren't replayed). $0.
- **What it is & Aim:** Agent A on each local model against rule-based buyers with known behaviour (offers its full budget at once; climbs from 60% of asking to its budget; offers exactly the floor). Every budget is above the floor, so a no-deal is the seller's fault. A free check before a paid tournament, written after v2 stopped closing.
- **Result (deals out of 10 vs full budget / climbing / floor; below-floor deals):** base 6 / 8 / 3, 8 of 17 below floor; SFT 3 / 10 / 1, 1 of 14; v2 0 / 3 / 0, 0 of 3; v3 9 / 7 / 10, 3 of 26 below floor (it accepted the climbing buyer's 60%-of-asking opening offer 3 times; see `finetune/DPO_V3.md` §6).
- **RQs Answered:** **RQ2** (diagnostic for v2 and v3).
- **Models:** Seller: `weak`, `finetuned`, `finetuned_dpo`, `finetuned_dpo_v3` (local Ollama). Buyers: `sanity:full_budget`, `sanity:linear`, `sanity:floor` (rule-based).

---

## 22. `attacks_dpo_v3` (`configs/attacks_dpo_v3.yaml`) — 400 games *(Done by ZY on 2026-11-10; DPO v3 training by Fayaz)*
- **Command:** `uv run python -m haggle.runner configs/attacks_dpo_v3.yaml`
- **Output Path:** `runs/attacks_dpo_v3/20261110-190057/` (scripted buyers and a local seller, so $0)
- **What it is & Aim:** The RQ3 attack scripts and their plain twins against DPO v3, on the same games as every other model. Gate 3 in `finetune/DPO_V3.md`: v3 now closes deals, so this checks it didn't become easy to talk into below-floor ones.
- **Result:** it did. 57 of 200 attacks won (SFT 14, v2 8, base 73, Haiku 0) and 14 twin failures, so attack minus twin is 43, worse than the untuned base model's 35. Agent A lost 49 of the 57, with or without defences; the worst categories were prompt injection (13), fake competing offers (12), fake urgency (10) and deadline pressure (10). v3 swapped v2's "never accepts" for "gives way under pressure"; neither compares the offer with the floor. Not taken to a paid tournament. v4 plan: `finetune/DPO_V3.md` §7.
- **RQs Answered:** **RQ3**, **RQ2** (v2 -> v3).
- **Models:**
  - Seller: `finetuned_dpo_v3` (`qwen2.5-3b-haggle-dpo-v3` on local Ollama)
  - Buyers: scripted attackers from `data/attacks.json`

---

## 23. `attacks_floorguard` (`configs/attacks_floorguard.yaml`) — 200 games *(Done by ZY on 2026-11-11)*
- **Command:** `uv run python -m haggle.runner configs/attacks_floorguard.yaml`
- **Output Path:** `runs/attacks_floorguard/20261111-144121/` (scripted buyers and local sellers, so $0)
- **What it is & Aim:** Every other run had the engine floor guardrail off, to measure the failure. This one turns it on
  (`floor_protection: true`) for Agent A on the base and fine-tuned models, on the same attack and twin games as
  `attacks`, `attacks_base` and `attacks_finetuned` (a test checks this), so the guardrail is the only difference.
- **Result (Agent A, off -> on):** below-floor deals base 34 -> 0, fine-tuned 3 -> 0. Attacks won base 32 -> 9 and
  fine-tuned 5 -> 3; the wins left are all floor leaks, which the price rule cannot stop. Leak games unchanged (base 31 ->
  32), so the re-ask does not leak. The base model hit the guardrail 60 times in 33 games and was forced to reject 11
  times. The run cannot show lost good deals: the scripts never offer above the floor.
- **RQs Answered:** **RQ3** and report Section 6 (does the code-level guardrail do what the prompt-level defences don't?).
- **Models:** Seller: `weak` and `finetuned` (local Ollama). Buyers: scripted attackers from `data/attacks.json`.
