# Fine-tuning (RQ2)

RQ2 asks whether a cheap local model, fine-tuned on real negotiations, can compete with a prompted hosted model.
We fine-tune **Qwen2.5-3B-Instruct** in two stages:

1. **SFT** (supervised fine-tuning with QLoRA) on seller turns from CraigslistBargain. This folder. The result is
   `qwen2.5-3b-haggle`, which the code and configs call "v1" or `finetuned`.
2. **DPO** (preference training) on top of the SFT model, in three versions (v2, v3, v4). See
   [`dpo/README.md`](dpo/README.md).

Agent C is the fine-tuned model with Agent A's prompt, and Agent D is the fine-tuned model in Agent B's loop. No agent
code is needed: both load from config once the model is registered in Ollama.

## Download the trained model

You don't need to retrain to run the experiments. **[Download the models here](LINK_TO_MODELS)** (the SFT model and
the three DPO versions), then go to step 3 below.

| File | SHA-256 |
|---|---|
| `qwen2.5-3b-haggle-q4_k_m.gguf` (SFT model, about 1.9 GB) | `2b57f5c2d5c62ab2a46d2d8a47f94e3b5c935baa71eb27ca1aba78edf0ed6175` |

The training log, loss curve and before/after check are committed in `runs/finetune/`; the write-up is
[`SFT_RESULTS.md`](SFT_RESULTS.md).

## Files

| File | What it does |
|---|---|
| `prepare_data.py` | Downloads CraigslistBargain and replays each dialogue through our engine, so training prompts match live prompts exactly |
| `download_base.py` | Downloads the pinned base weights for the MLX route |
| `mlx_lora.yaml` | Training settings for the MLX route |
| `run_train.sh` | Runs the MLX training and logs it to `runs/finetune/train.log` |
| `qlora_colab.ipynb` | The same training on any CUDA GPU (Colab or local Jupyter), with resume, scoring and GGUF export |
| `plot_loss.py` | Plots the loss curve from an MLX log and names the checkpoint with the lowest held-out loss |
| `Modelfile` | Serves the GGUF through Ollama with the stock Qwen chat template |
| `compare_models.py` | Compares two Ollama models on 100 held-out seller turns |

There are two routes to the same training: **MLX** (Apple-silicon Macs only), or the **notebook** (any CUDA GPU,
including Colab). The model in the report was trained with the MLX route.

## Before you start

You need `uv`, Ollama, and about 25 GB of free disk. For the GGUF export on the MLX route you also need `cmake` and a
C++ compiler.

```bash
uv sync
ollama pull qwen2.5:3b
mkdir -p runs/finetune
```

## 1. Build the training data

```bash
uv run python -m finetune.prepare_data
```

This downloads CraigslistBargain into `data/craigslist/` (git-ignored) and writes `data/craigslist/sft/train.jsonl`
and `valid.jsonl`. You should get 3,141 training and 337 held-out examples, with these SHA-256 checksums:

```text
6bb28826ebb4e55d269ef6f9b5fef5c395bd820142cc2209d8371c8bc1e59ad7  train.jsonl
40752a2545f89b2a3aa2c760f7a3111efb0d2f94a3289c0aa29f996d9ec18ea2  valid.jsonl
```

## 2. Train

### Route A: notebook (any CUDA GPU)

1. Open `finetune/qlora_colab.ipynb` in Colab (Runtime > Change runtime type > pick a GPU) or in local Jupyter.
2. Run all. On Colab it mounts Google Drive and asks you to upload `train.jsonl` and `valid.jsonl` from
   `data/craigslist/sft/`.
3. If the session drops, run all again. It resumes from the last checkpoint on Drive.
4. It writes `training_log.json`, `loss_curve.png`, `report.json` and the GGUF to `MyDrive/haggle_finetune` (or
   `out/` when not on Colab). Move the GGUF to `runs/finetune/`, then go to step 3.

Optional environment variables, read by the notebook's setup cell:

| Variable | Default | Meaning |
|---|---|---|
| `HAGGLE_DATA_DIR` | `.` | Folder holding `train.jsonl` and `valid.jsonl` |
| `HAGGLE_OUT_DIR` | `MyDrive/haggle_finetune` on Colab, else `out` | Where checkpoints and outputs go |
| `HAGGLE_MAX_STEPS` | `-1` (full training) | Set a small number for a smoke test |
| `HAGGLE_BATCH_SIZE` | `4` | Lower it if you run out of GPU memory; gradient accumulation keeps the effective batch at 8 |
| `HAGGLE_EVAL_EVERY` | `50` | Optimizer steps between held-out checks |
| `HAGGLE_EVAL_EXAMPLES` | `100` | Held-out examples scored at each check |
| `HAGGLE_N_CHECK` | `100` | Held-out examples in the before/after check |

### Route B: MLX (Apple-silicon Macs)

Download the pinned base weights, then train with the settings in `finetune/mlx_lora.yaml`:

```bash
uv run --no-project --with huggingface_hub python finetune/download_base.py
bash finetune/run_train.sh        # or: uv run --with mlx-lm==0.32.2 mlx_lm.lora --config finetune/mlx_lora.yaml 2>&1 | tee runs/finetune/train.log
```

Checkpoints are saved every 100 iterations. To resume from, say, iteration 800, add
`--resume-adapter-file runs/finetune/adapters/0000800_adapters.safetensors --iters 770` and use `tee -a`. The
learning-rate schedule restarts when you resume.

If it runs out of memory, set `batch_size: 2`, `grad_accumulation_steps: 4`, `iters: 3141` and
`arguments: [2.0e-4, 3141, 1.0e-6]` in the YAML. This is the same training, just slower.

**What healthy looks like:** training and held-out loss both fall, then level off. If held-out loss climbs while
training loss keeps falling, it's overfitting: you can stop early, because the next step picks the best checkpoint.

Pick the checkpoint with the lowest held-out loss (MLX's last checkpoint isn't necessarily its best):

```bash
uv run --group analysis python -m finetune.plot_loss runs/finetune/train.log
cp runs/finetune/adapters/<the file it names> runs/finetune/adapters/adapters.safetensors
```

Then fuse the adapter into the full-precision weights and convert to Q4_K_M, the same format as stock `qwen2.5:3b`:

```bash
uv run --with mlx-lm==0.32.2 mlx_lm.fuse --model runs/finetune/base-bf16 --adapter-path runs/finetune/adapters --save-path runs/finetune/fused

git clone --branch b6000 --depth 1 https://github.com/ggml-org/llama.cpp runs/finetune/llama.cpp
uv run --no-project --with torch==2.6.0 --with transformers==4.51.3 --with numpy --with sentencepiece --with protobuf --with safetensors \
  python runs/finetune/llama.cpp/convert_hf_to_gguf.py runs/finetune/fused --outtype f16 --outfile runs/finetune/fused-f16.gguf

cmake -S runs/finetune/llama.cpp -B runs/finetune/llama.cpp/build -DLLAMA_CURL=OFF
cmake --build runs/finetune/llama.cpp/build --target llama-quantize -j
runs/finetune/llama.cpp/build/bin/llama-quantize runs/finetune/fused-f16.gguf runs/finetune/qwen2.5-3b-haggle-q4_k_m.gguf q4_k_m
```

Expect a GGUF of about 1.9 GB. You can delete `fused/` and `fused-f16.gguf` afterwards. (Ollama can't import Qwen
from safetensors itself, hence llama.cpp.)

## 3. Serve and check it

With the GGUF (downloaded or trained) in `runs/finetune/`:

```bash
cp finetune/Modelfile runs/finetune/Modelfile
(cd runs/finetune && ollama create qwen2.5-3b-haggle -f Modelfile)
uv run python -m finetune.compare_models qwen2.5:3b qwen2.5-3b-haggle --out runs/finetune/report.json
```

`compare_models` scores both models on 100 held-out seller turns, without forcing the JSON format:

| In `report.json` | Stock `qwen2.5:3b` | Fine-tuned (report) |
|---|---|---|
| `valid_json_rate` | 0.92 | 0.99 |
| `same_action_as_human` | 0.52 | 0.72 |
| `offers_below_floor` | 0.18 | 0.13 |
| `median_price_error` | 0.125 | 0.014 |

Then run the RQ2 experiments from the main README (`configs/tournament_finetuned.yaml`,
`configs/attacks_finetuned.yaml`, `configs/tournament_finetuned_ablation.yaml`).

## The model used in the report

Trained with the MLX route and the settings in `mlx_lora.yaml` (LoRA rank 16 on q/k/v/o, learning rate 2e-4,
effective batch 8, seed 42). Training stopped at iteration 430 of the planned 1,570 (about half a pass over the data).
We used the checkpoint at iteration 300, which had the lowest held-out loss (0.975).

## Why the data looks the way it does

- **Which dialogues:** of 5,247 training dialogues, 671 were kept: 580 good deals and 91 refusals of offers below
  the floor. `data/craigslist/sft/funnel.json` shows why the rest were dropped.
- **Floors are synthetic:** the dataset has none, so one is drawn the way `make_scenario` does, then adjusted to fit
  what the seller did.
- **`wait_hours` is always 0:** the dataset has no usable timing, so the model won't learn pacing.
- **No condition facts:** the prompt's facts section is a neutral placeholder, so this data can't teach honesty.
- **The published test split isn't used:** it has no price annotations. The held-out set is the dev split.

## Licences

- **Qwen2.5-3B** is under the Qwen Research Licence, so research use only.
- **CraigslistBargain** lists its licence as unknown on Hugging Face. Cite He et al. (2018), and don't redistribute it.
