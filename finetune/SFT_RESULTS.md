# SFT training and evaluation report (RQ2)


**Trained Model:** `qwen2.5-3b-haggle` (Base: `mlx-community/Qwen2.5-3B-Instruct-4bit`, 4-bit QLoRA, rank 16)
**Run Time:** MLX route (`finetune/mlx_lora.yaml`); ran 430 of 1,570 planned iterations in **2 hours 10 minutes** (20:34 to 22:43 on 5 Nov 2026). Stopped early after validation loss rose at step 400 (`0.986` vs `0.975` at step 300).
**Selected Checkpoint:** Checkpoint 300 (`0000300_adapters.safetensors`, validation loss: **0.975**)
**Artifact Format:** GGUF `Q4_K_M` (1.83 GB, SHA-256 `2b57f5c2d5c62ab2a46d2d8a47f94e3b5c935baa71eb27ca1aba78edf0ed6175`)

---

## 1. Quantitative Evaluation Results

Evaluation performed on **100 held-out seller turns** sampled from the **CraigslistBargain `dev` split** (`data/craigslist/sft/valid.jsonl`, 337 seller turns from 77 kept dialogues; the published `test` split has no price/intent annotations and is not used) via `finetune.compare_models`:

| Evaluation Metric                | Stock Baseline (`qwen2.5:3b`) | Fine-Tuned (`qwen2.5-3b-haggle`) | Net Difference                                                                 |
| -------------------------------- | ------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------ |
| **Valid JSON Format Rate** | 92.0%                           | **99.0%**                    | **+7.0 pts** (parsed as valid `Action` without schema enforcement)       |
| **Human Action Alignment** | 52.2%                           | **71.7%**                    | **+19.5 pts** (matches the human seller's action type among valid JSONs) |
| **Offers Below Floor**     | 18.2%                           | **13.0%**                    | **-5.2 pts** (share of model `offer` moves below the synthetic floor)    |
| **Median Price Error**     | 12.5%                           | **1.4%**                     | **-11.1 pts** (`|model_price - human_price| / human_price` on shared offers) |
| **Tokens per Second**      | 32.5 tok/s                      | **26.2 tok/s**               | Real-time interactive generation preserved                                     |
| **Latency per Response**   | 1.37s                           | **1.48s**                    | Negligible overhead                                                            |

### Key Findings for the Final Report:

1. **Schema Compliance:** Without JSON-schema constrained decoding, the fine-tuned model outputs valid `Action` JSON (`message`, `action`, `price`, `wait_hours`) on 99.0% of turns vs. 92.0% for stock `qwen2.5:3b`.
2. **Offer Price Closeness to Human Sellers (`median_price_error`):** When both the model and the recorded human seller make an `offer`, `median_price_error` measures `|reply.price - truth.price| / truth.price`. Fine-tuning reduced the median gap between the model's counter-offer and the human seller's actual offer from 12.5% to 1.4%.
3. **Below-Floor Offers Remain a Warning Sign (`offers_below_floor`):** Because CraigslistBargain has no reservation prices, floors in `prepare_data.py` are synthetic (drawn at 70–90% of asking price and capped so the human seller's moves never cross them). While fine-tuning reduced below-floor offers from 18.2% to 13.0% of `offer` moves, **13.0% is still high** (roughly 1 in 8 offers violates the floor), showing that SFT alone does not guarantee floor discipline and deterministic engine/agent guardrails remain essential.
4. **Action Selection (`same_action_as_human`):** 71.7% of valid replies chose the same move type (`offer`, `accept`, `chat`, `reject`, `walk_away`) as the reward-filtered human seller (up from 52.2%).

---

## 2. Where to Put the Downloaded Files

Download the files (links in `finetune/README.md`) and place them in the repository under `runs/finetune/`.

### Exact Directory Structure:

```text
carouhaggler-demo/
├── configs/
│   ├── tournament_finetuned.yaml
│   └── attacks_finetuned.yaml
├── data/
│   └── craigslist/
│       └── sft/
│           ├── train.jsonl
│           └── valid.jsonl
├── runs/
│   └── finetune/                   <-- PUT THE DOWNLOADED FILES HERE (links in finetune/README.md)
│       ├── Modelfile               <-- Ollama registration file
│       ├── qwen2.5-3b-haggle-q4_k_m.gguf  <-- 1.83 GB quantized model
│       ├── loss_curve.png          <-- Plot for final report
│       ├── report.json             <-- 100-turn evaluation metrics
│       ├── train.log               <-- Training terminal output
│       └── training_log.json       <-- Loss data per checkpoint
└── finetune/SFT_RESULTS.md
```

### Destination Summary Table:

| File Name                             | Destination Path in Repo                        | What It Is Used For                                         |
| ------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------- |
| `qwen2.5-3b-haggle-q4_k_m.gguf`     | `runs/finetune/qwen2.5-3b-haggle-q4_k_m.gguf` | Quantized model loaded by Ollama                            |
| `Modelfile`                         | `runs/finetune/Modelfile`                     | Ollama blueprint (`FROM ./qwen2.5-3b-haggle-q4_k_m.gguf`) |
| `loss_curve.png`                    | `runs/finetune/loss_curve.png`                | Training vs Validation loss figure for paper                |
| `report.json`                       | `runs/finetune/report.json`                   | Quantitative evaluation numbers for paper table             |
| `train.log` & `training_log.json` | `runs/finetune/`                              | Training logs verifying convergence at step 300             |

---

## 3. Setup Guide (Step-by-Step)

### Step 1: Ensure Ollama is Running & Pull Stock Baseline

```bash
# In any terminal
ollama pull qwen2.5:3b
```

### Step 2: Register the Fine-Tuned Model into Ollama

You **must** be inside `runs/finetune/` when running `ollama create` because `Modelfile` references `./qwen2.5-3b-haggle-q4_k_m.gguf`:

```bash
# 1. Navigate to the folder containing the Modelfile and GGUF
cd runs/finetune

# 2. Register the model in Ollama
ollama create qwen2.5-3b-haggle -f Modelfile

# 3. Return to project root
cd ../..
```

### Step 3: Verify Both Models are Registered

```bash
ollama list
```

You should see:

- `qwen2.5:3b`
- `qwen2.5-3b-haggle:latest`

---

## 4. How to Run the Experiments

All evaluation commands run from the **project root** directory using `uv`:

### A. Reproduce the 100-Turn Evaluation Benchmark (~3 mins)

Validates JSON formatting, pricing errors, and human alignment:

```bash
uv run python -m finetune.compare_models qwen2.5:3b qwen2.5-3b-haggle --out runs/finetune/report.json
```

### B. Run Full RQ2 Tournament (Agents C & D in Live Play)

Runs the fine-tuned model as **Agent C** (prompted) and **Agent D** (agentic loop with tools) across 20 Singapore listings against tournament buyers:

```bash
uv run python -m haggle.runner configs/tournament_finetuned.yaml
```

### C. Run RQ2 + RQ3 Red-Team Attack Matrix

Evaluates Agents C & D against the 50 adversarial attacks (sob stories, prompt injections, floor extraction) and neutral twins:

```bash
uv run python -m haggle.runner configs/attacks_finetuned.yaml
```

### D. Test Interactively in the Streamlit UI

Play live against your fine-tuned model in the browser:

```bash
uv run streamlit run app/demo.py
```

*(Select `qwen2.5-3b-haggle` from the model dropdown).*
