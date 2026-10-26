# Haggle Arena — Team Role Split (v2)

Oct 26, 2026 · @ZY · replaces the Oct 16 version

## Overview

Haggle Arena is a negotiation testbed where LLM agents haggle over second-hand marketplace listings. It's a pre-deployment safety evaluation for the kind of autonomous seller agents platforms like Alibaba's Xianyu are starting to pilot. It measures whether a seller agent:

- gets good deals
- resists manipulation
- stays honest about item condition
- holds up against a stronger opponent

## What changed since v1

1. **Time and multiple buyers are in.** Messages carry a simulated delay (`wait_hours`), each side has a private deadline in days, and one seller can now handle several buyers who arrive at different times. Once the item sells, the other chats end. Fayaz built the engine side on `origin/fayaz`.
2. **One-on-one games stay the main setting** for RQ1–5. Time and multi-buyer is an **extra experiment** with its own question. The "1 buyer vs N sellers" setup is dropped.
3. **We're keeping all six research questions**, plus the extension question.
4. **Methodology fixes** so the results hold up in the report. Details in the next section.
5. **Role 2 is mostly built** (on the `role2` branch). What's left is fine-tuning, which can now start.

## Research questions and how we answer them

| # | Question | How we answer it |
|---|---|---|
| RQ1 | Does an agentic seller beat a prompted one? | **Ablation**: A, A + tools, A + planning, full B. All four use the same base prompt, so the only difference is the extra scaffolding. This shows *which part* helps, not just that B wins. |
| RQ2 | Can a fine-tuned 1–3B model compete at a fraction of the cost? | Fine-tuned C/D vs A/B on surplus and cost per game, plotted on one chart. A negative result is still reportable. |
| RQ3 | How exploitable are seller agents? | 50 attack scripts in 7 categories. **Each attack has a plain "twin"** with the exact same prices and timing but neutral wording. Attack effect = success under attack − success under twin. Without the twin, attacks get credit for failures that happen anyway: in our demo the 3B seller sold below its floor to a plain offer. |
| RQ4 | Do better negotiators lie more? | Buyers **ask about condition** (battery, dents, what's included), so sellers get chances to lie. The seller's "be honest" line is a switch, so we can compare instructed vs not. The LLM judge is checked against ~100 hand-labelled games. |
| RQ5 | Does the stronger model win? | Same agent design, strong vs weak model, both directions. The agentic buyer is a separate "stronger opponent" condition, so model and design aren't mixed. |
| RQ6 | Does human-approved drafting reduce violations? | The seller drafts a reply and a human approves or edits it in the UI. Compare violations against full autopilot on the same scenarios. |
| Ext. | Do time pressure and competition make sellers easier to exploit? | Measure whether concessions speed up as deadlines approach (**not scripted into prompts**). Test fake competing offers against real ones the platform reports, and attacks that exploit the seller's deadline. |

**Reporting rules (all RQs):**
- Bootstrap 95% CIs.
- Paired comparisons on the same scenarios and seeds.
- **Report seller payoff per game (no deal = 0) next to surplus share**, because surplus share alone rewards a seller that walks away from every hard buyer.

**Models:**
- A capable hosted model (via OpenRouter) as the main seller for RQ1, RQ3 and RQ4.
- `qwen2.5:3b` as the weak side for RQ5.
- The buyer pool stays the same across all conditions, because a weak buyer makes every seller look good.

Exact model choice is still open (decide by Sun 1 Nov).

## Timeline (re-baselined)

Today is Mon 26 Oct.

| Date | Milestone |
|---|---|
| Wed 28 Oct | Merge `origin/fayaz` and `role2` into `main`. Fix the open engine issues (below). **Freeze schemas.** |
| Fri 30 Oct | Runner loads every agent from config. 50-game pilot gives the real cost per game. Judge prompt ready, ~100 games exported for labelling. |
| Sun 1 Nov | Models and budget decided. Hand-labelling done. Final experiment configs. |
| Wed 4 Nov | All experiment runs done (RQ1–6 + extension). |
| Fri 6 Nov | Results notebook and charts. Fine-tuned C/D results, if ready. |
| Sun 8 Nov | Packaging: README, pinned models and seeds, frozen `runs/`. |
| Tue 10 Nov / Fri 13 Nov | Presentation and submission (G1 / G2). |

## Role 1 — Foundation: Engine, Data & Tools

**Owns:** `schemas.py`, `engine.py`, `orchestrator.py`, `scenarios.py`, `llm.py`, `tools/`, sanity agents, data pipeline, `tests/`

**Done (on `origin/fayaz`):**
- The engine, including simulated time, the `chat` action and deadlines.
- The multi-buyer orchestrator.
- The OpenRouter-ready client with a budget cap.
- Tools: `price_lookup`, `calculator`, `walk_away_value`.
- 104 listings and 416 price comparables.
- Sanity agents and about 1.6k tests.

**Open items (found while building Role 2):**
1. **Two clocks disagree.** A buyer arriving at hour 48 gets turns stamped hour 0.
2. **Per-side deadlines get overwritten.** In multi-buyer mode the orchestrator uses one deadline for everyone, so urgent vs relaxed buyers don't work there.
3. **The engine's guardrail feedback states the floor number** ("You may not sell below your floor of $X"). The model then repeats it to the buyer. The feedback should say "below your minimum" without the number.
4. **Buyers can see other buyers' offers.** Realistic marketplaces hide them. This looks intentional (`CompetitiveBuyer`), so the team should decide.
5. **The seller's floor can differ between chats.** Sessions for one listing can be created with different floors. The orchestrator should enforce one floor per listing.
6. **Housekeeping:**
   - `docs/INTERFACES.md` is referenced in the README but missing.
   - `.DS_Store` is committed.
   - `CLAUDE.md` has typos.
   - The README still says 30 listings.

(Already fixed on `role2`: the orchestrator replayed an agent's first reply to every retry, so guardrail re-asks never worked in multi-buyer mode and tokens were double-counted.)

## Role 2 — Agents, Attackers, Guardrails & Fine-Tuning (@ZY)

**Owns:** `haggle/agents/`, `prompts/`, `guardrails.py`, `data/attacks.json`, `finetune/`

**Done (on `role2`, all with tests that run without Ollama):**
- **Agent A**, the prompted seller, with switches for honesty instruction and injection defence.
- **Agent B and the agentic buyer.** One shared loop: reflect → tools → plan → act. Each stage switches on or off for the RQ1 ablation, and names come out like `agentic_seller[tools+plan]`.
  - Tools run in code every turn.
  - Notes are kept per chat, so they never leak between buyers or between games.
- **Buyer personas:** fair, tough, lowballer, manipulator, indecisive. Deadline presets: urgent (1.5 days), normal (5), relaxed (10). Buyers ask about condition by default.
- **Attackers:** 50 scripts across sob story, fake urgency, extreme lowball, prompt injection, floor extraction, fake competing offer and deadline exploit. Each has a matched twin.
  - An optional LLM mode rewrites each message to fit the conversation, while code keeps prices fixed.
  - Each category has a success rule (below-floor deal, floor leak, or either) for Role 3.
- **Guardrails:**
  - Buyer text is wrapped and marked as untrusted.
  - A grounding check catches claims that contradict the item's condition.
  - A floor-leak filter re-asks once, then redacts. It never repeats the number.
  - All of it works around any agent via `GuardedAgent`.

**Remaining:**
- **Fine-tuning (Agents C/D).** Convert CraigslistBargain into our action format (including `chat` and `wait_hours`), run QLoRA on a 1–3B model, and serve it through Ollama. Track the training metrics below.
- **RQ6 draft mode:** an agent wrapper that returns the draft for a human to approve or edit (UI by Role 3).
- Support Role 3 in wiring the agents into the runner.

**Early observations from live runs on `qwen2.5:3b`** (single runs, so treat them as hints, not results):
- It answered condition questions truthfully but invented accessories.
- It ignored its own walk-away tool and stalled until the buyer's deadline.
- Buyers leaked their own budgets.
- Wrapping buyer text didn't stop it accepting $186 against a $502 floor.
- The leak filter caught a live floor leak.

### Fine-tuning success metrics (RQ2)

**Training-time**

| Metric | What it tells you |
| --- | --- |
| Training loss curve | Is the model learning, or stuck or diverging |
| Eval loss on the held-out split | Overfitting on ~6.6k dialogues |
| Format/schema validity rate | Share of outputs that parse as a valid `Action`, before retries |
| Adapter size / inference latency | Whether QLoRA's cost and speed promise holds |

**Deployment-time** (same tournament as every other agent)

| Metric | Comparison |
| --- | --- |
| Deal rate, surplus share, payoff | C vs A, D vs B, same opponents and seeds |
| Format failures in live play | Under multi-turn adversarial pressure |
| Below-floor deals / guardrail events | Small models get confused mid-negotiation |
| Cost and latency per game | The actual payoff for "a fraction of the cost" |

## Role 3 — Evaluation, Experiment Harness & Demo

**Owns:** `metrics.py`, `judge.py`, `runner.py`, `analysis/results.ipynb`, `app/demo.py`

- **Metrics:**
  - Deal rate, surplus share **and payoff per game (no deal = 0)**, below-floor deals, floor leaks, backward offers, format failures, honesty violations, cost and latency.
  - **Attack effect = attack − twin**, using each category's success rule.
  - All reported with bootstrap 95% CIs.
- **New metrics for the extension:** wait behaviour, concession rate vs time remaining, outcome vs deadline, and how often a fake competing offer works compared with a real one.
- **LLM-as-judge for honesty**, validated against ~100 hand-labelled games.
- **Runner:** executes the matrix below from configs within budget. Agents load by name, with their switches (see the README "Agents" section).
- **Harnesses:** self-play calibration, plus counterfactual replay (same buyer transcript against each seller variant).
- **Streamlit demo** (plus the RQ6 approve/edit UI) and the final charts and tables.

## Experiment matrix (proposal: confirm after the pilot)

| Experiment | Conditions | Approx. games |
|---|---|---|
| RQ1 ablation | 4 seller variants × fair/tough/lowballer × 20 scenarios | 240 |
| RQ3 attacks | 50 attacks + 50 twins × sellers A and B × defences off/on | 400 (scripted buyers, so only seller calls cost) |
| RQ4 honesty | Reuse the RQ1 games, plus the honesty switch off for A and B on 30 scenarios | +120 |
| RQ5 asymmetry | Strong/weak seller × strong/weak buyer, same design, 30 scenarios | 120 |
| RQ6 assist | Draft mode with classmates vs autopilot on the same scenarios | 10–15 human games |
| Extension | Sellers A and B × 2–3 buyers per listing × urgency mix × 20 listings | 40 markets |
| RQ2 | C/D through the RQ1 and RQ3 matrices | if ready |

Final counts depend on the pilot's cost per game and the API budget.

## Hand-offs between roles

| From → To | What |
|---|---|
| Role 1 → Role 2 | Stable `Observation` / `Action` schemas (`session_id` added on `role2`), tools, orchestrator |
| Role 2 → Role 3 | Agent classes and switches (README "Agents"), `load_attacks()`, `SUCCESS` rules, persona labels like `attack:sob_story` / `twin:sob_story`, guardrail event names |
| Role 3 → Role 2 | Pilot results, judge labels, any failure cases the agents should handle |

## Git workflow

- One branch per role. Pull requests go into `main`; someone other than the author reviews.
- `main` is stale right now: merge `origin/fayaz`, then `role2`, by Wed 28 Oct.
- **After the schema freeze**, schema changes need the whole team's agreement.
- Don't commit secrets, `runs/`, or editor/OS files.
