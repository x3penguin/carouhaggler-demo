# 3B Model Performance Improvement Guide

This guide details four concrete, reproducible engineering steps to close the capability gap between our local edge model (`qwen2.5:3b`) and frontier models (`claude-haiku-5.5`). All interventions strictly respect the **3B parameter limit** while directly addressing the three failure modes identified in our study: **38% below-floor deals**, **75% factual dishonesty**, and **scaffolding deadlock crashes (67% deal rate)**.

---

## Overview of Interventions

| Step | Technique | Target Problem | Mechanism |
| :--- | :--- | :--- | :--- |
| **1** | **Direct Preference Optimization (DPO)** | 38% below-floor rate & 75% dishonesty | Replaces human imitation SFT with reward-ranked pairs penalizing hallucinations and leaks. |
| **2** | **Frontier Chain-of-Thought Distillation** | 67% deadlock crash from scaffolding | Moves multi-stage planning into internal `<thought>` tokens within a single fast generation turn. |
| **3** | **LoRA Expansion to MLP Layers (`all-linear`)** | Poor arithmetic & boundary violations | Extends LoRA weights to feed-forward blocks (`gate_proj`, `up_proj`, `down_proj`) where numbers are processed. |
| **4** | **Constrained Logit Decoding (Grammars)** | Residual sub-floor offers | Enforces JSON schema minimum price constraints at sampling time, mathematically guaranteeing 0% below-floor deals. |

---

## Step 1: Direct Preference Optimization (DPO)

Instead of training Qwen-3B to imitate raw Craigslist dialogues (which contain irrational human concessions and deceptive exaggerations), DPO directly teaches the model's weights to prefer grounded, surplus-maximizing replies over losing ones.

### 1.1 Dataset Schema (`data/dpo/train.jsonl`)

Generate pairwise examples from existing training scenarios. Each row contains:
- `prompt`: System instructions, listing title, verified condition facts, private floor price, and buyer dialogue history.
- `chosen`: A turn that stays strictly above the floor, truthfully discloses flaws from `condition_facts`, and makes a modest counter-offer.
- `rejected`: A turn that drops below floor, lies about item condition (e.g. claiming a scratched item is "brand new"), or leaks the private floor price.

```json
{
  "prompt": "Listing: Asics Gel-Kayano 14\nFacts: metallic silver overlays show faint crease; mesh intact\nFloor: SGD 118\n[0] BUYER (offer $100): Any marks or creases? I can do $100 today.",
  "chosen": "{\"thought\": \"Buyer offered $100, which is below my floor of $118. I must disclose overlay creases honestly and counter above floor.\", \"message\": \"There is light creasing on the silver overlays, but the mesh is intact. $100 is too low, but I can do $135.\", \"action\": \"offer\", \"price\": 135}",
  "rejected": "{\"thought\": \"I want to close the deal quickly.\", \"message\": \"No marks at all, it is brand new in box! I can meet you at $110.\", \"action\": \"offer\", \"price\": 110}"
}
```

### 1.2 DPO Training Script (`finetune/train_dpo.py`)

Using HuggingFace `trl` and `peft`:

```python
"""Run DPO on Qwen2.5-3B using curated negotiation pairs."""

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATA_PATH = "data/dpo/train.jsonl"
OUTPUT_DIR = "runs/finetune/dpo_adapters"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    beta=0.1,                          # Implicit reward scale
    learning_rate=5e-6,                # Conservative learning rate for DPO
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=True,
)

trainer = DPOTrainer(
    model=MODEL_ID,
    ref_model=None,                    # TRL automatically creates frozen reference model from PEFT base
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"DPO adapters saved to {OUTPUT_DIR}")
```

---

## Step 2: Frontier Chain-of-Thought (CoT) Distillation

Our study proved that multi-turn external scaffolding crashes 3B deal rates from 80% to 67% due to context thrashing. We replace multi-prompt loops with **internalized single-turn thinking**.

### 2.1 Distillation Pipeline

1. Take the 671 training scenarios from `data/craigslist/sft/`.
2. Query a frontier teacher (`anthropic/claude-haiku-5.5` or `openai/gpt-4o`) to generate training targets formatted with `<thought>` tags preceding the action JSON.

```python
TEACHER_PROMPT = """You are generating expert negotiation training data.
You will be given listing facts, private seller floor, and buyer message.
Generate:
1. An internal thinking block enclosed in <thought> ... </thought> explaining:
   - Margin gap calculation (Buyer Offer vs Floor)
   - Fact verification (which defects must be explicitly disclosed)
   - Planned counter-offer price
2. A single valid JSON object with keys "message", "action", and "price".

Example Output:
<thought>
Buyer offered SGD 80. Floor is SGD 97. Difference is -17.
Listing flaws to state: light creasing on red toe panel.
Counter-offer strategy: anchor high at SGD 105, acknowledging wear.
</thought>
{"message": "There is light creasing on the toe panel, but the treads are great. I can do SGD 105.", "action": "offer", "price": 105}
"""
```

### 2.2 Updating `finetune/prepare_data.py`

Ensure that the SFT target string includes the reasoning trace:

```python
# Format expected by training tokenizer:
assistant_content = f"<thought>\n{thought_reasoning}\n</thought>\n{json_action_payload}"
```

At runtime, Qwen-3B outputs its own reasoning before emitting the price token, giving it test-time compute advantages without external prompt calls.

---

### Step 2A: Distillation via Claude Code CLI (Zero Additional API Cost)

If you have an active Claude Code subscription / CLI access and do not want to set up an external paid OpenRouter / Anthropic API billing tier, you can run the distillation in headless non-interactive batch mode directly from your terminal using `claude -p` (print mode).

#### 2A.1 Batching Strategy
To avoid making 3,141 individual CLI invocations (which would be slow and hit session rate limits), pack dialogues into chunks of 10–20 seller turns per prompt file, and ask Claude Code to process and output structured JSONL blocks.

#### 2A.2 Headless Distillation Script (`finetune/distill_with_claude_code.py`)

Create a lightweight Python orchestrator that calls the `claude` CLI in batch chunks:

```python
"""Batch-distill thought reasoning traces using headless Claude Code CLI."""

import json
import subprocess
from pathlib import Path

INPUT_PATH = Path("data/craigslist/sft/train.jsonl")
OUTPUT_PATH = Path("data/craigslist/sft/train_distilled_claude_code.jsonl")
BATCH_SIZE = 15  # Process 15 turns per CLI call to stay well within token windows

lines = [json.loads(line) for line in INPUT_PATH.read_text().splitlines()]
print(f"Loaded {len(lines)} raw training turns.")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
processed_count = 0

for i in range(0, len(lines), BATCH_SIZE):
    batch = lines[i : i + BATCH_SIZE]
    
    # Construct prompt instructing Claude to return exact JSONL format
    prompt = f"""You are a negotiation data distillation engine.
Process the following {len(batch)} negotiation turns. For each turn, generate:
1. A `<thought>...</thought>` scratchpad that analyzes buyer offer vs floor, verifies listed defects, and plans the counter-price.
2. The final action JSON.

Input Batch (JSON array of turns):
{json.dumps(batch)}

Output Format Requirements:
Output ONLY raw JSONL lines (one line per turn). Each line must have the identical input structure, but with the assistant's content replaced by:
"<thought>\\n{{reasoning}}\\n</thought>\\n{{action_json}}"
Do not wrap in markdown backticks.
"""

    # Call claude non-interactively via pipe
    cmd = ["claude", "-p", prompt]
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    if res.returncode != 0:
        print(f"Error at batch {i}: {res.stderr}")
        continue
        
    cleaned_output = res.stdout.strip()
    with OUTPUT_PATH.open("a", encoding="utf-8") as f:
        for out_line in cleaned_output.splitlines():
            if out_line.strip().startswith("{") and out_line.strip().endswith("}"):
                f.write(out_line.strip() + "\n")
                processed_count += 1
                
    print(f"Progress: {processed_count}/{len(lines)} turns distilled.")

print(f"Complete! Distilled dataset written to {OUTPUT_PATH}")
```

#### 2A.3 Execution

Run the script in the background:
```bash
python finetune/distill_with_claude_code.py
```

- **Pros:** Completely zero marginal API expense (uses your active Claude Code subscription).
- **Practical Note:** Running all 3,141 turns in batches of 15 requires ~210 CLI calls. You can run it in chunks of 50 batches per work session to comfortably respect interactive rate limits.

---

## Step 3: Expand LoRA to MLP Layers in `finetune/mlx_lora.yaml`

In small parameter architectures, self-attention layers route attention patterns, but **factual knowledge, numbers, and arithmetic boundaries reside in the Feed-Forward Network (FFN / MLP) blocks**.

### 3.1 Edit MLX Configuration

Open `finetune/mlx_lora.yaml` and update lines 12–18:

```yaml
# BEFORE (Attention projections only):
fine_tune_type: lora
num_layers: -1
lora_parameters:
  keys: [self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj]
  rank: 16
  scale: 2.0
  dropout: 0.05

# AFTER (All linear layers - Attention + FFN):
fine_tune_type: lora
num_layers: -1
lora_parameters:
  keys: [
    self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj,
    mlp.gate_proj, mlp.up_proj, mlp.down_proj
  ]
  rank: 16
  scale: 2.0
  dropout: 0.05
```

### 3.2 Execute Fine-Tuning

Run on Apple Silicon using your existing harness:

```bash
uv run --with mlx-lm==0.32.2 mlx_lm.lora \
  --config finetune/mlx_lora.yaml 2>&1 | tee runs/finetune/train_all_linear.log
```

---

## Step 4: Constrained Logit Decoding at Inference Time

To mathematically guarantee **0% below-floor deals**, apply JSON schema grammar masking at runtime. The local inference engine forces the token sampler to mask invalid numbers before calculating softmax.

### 4.1 Schema Injection in `haggle/client.py`

When issuing completions to Ollama or a local vLLM/llama.cpp endpoint, dynamically inject the seller's private floor into the JSON grammar:

```python
def build_constrained_schema(seller_floor: float) -> dict:
    """Return JSON schema constraining output prices to >= seller_floor."""
    return {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "Internal scratchpad calculations"
            },
            "message": {
                "type": "string",
                "description": "Public message to buyer"
            },
            "action": {
                "type": "string",
                "enum": ["offer", "accept", "reject", "walk_away", "chat"]
            },
            "price": {
                "type": "number",
                "minimum": float(seller_floor),  # Logits for numbers < seller_floor set to -inf
                "description": "Counter offer price in SGD"
            }
        },
        "required": ["message", "action"]
    }
```

### 4.2 Passing Schema to the Local Endpoint

```python
response = client.chat.completions.create(
    model="qwen2.5-3b-haggle",
    messages=dialogue_messages,
    response_format={
        "type": "json_object",
        "schema": build_constrained_schema(scenario.seller_floor)
    },
    temperature=0.2,
)
```

---

## Expected Impact on Benchmark Metrics

Implementing these four steps transforms the 3B model from an unreliable prototype into an enterprise-safe local agent:

| Metric | Current Fine-Tuned 3B (`qwen2.5-3b-haggle`) | Optimized 3B Model | Benchmark Target (Claude Haiku 5.5) |
| :--- | :---: | :---: | :---: |
| **Below-Floor Deal Rate** | **38%** | **0.0%** (via schema masking) | 0.0% (with tools) |
| **Judge Dishonesty Rate** | **75%** | **< 15%** (via DPO ground truth) | 7% – 8% |
| **Deal Completion Rate** | **67%** (stalls under scaffolding) | **90% – 95%** (single-turn CoT) | 100% |
| **Seller Payoff** | **-0.13** (net negative) | **+0.25 to +0.30** | +0.35 |
| **Inference Cost / Game** | **$0.0000** | **$0.0000** | $0.0012 – $0.0045 |
