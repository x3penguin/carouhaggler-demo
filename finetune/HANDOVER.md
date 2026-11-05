# Fine-tuning handover (RQ2): for Fayaz

**Your job:** train the fine-tuned seller on your M2 Pro, serve it through Ollama, check it, and send the
results back to ZY. You don't need to write any code.

**Time:** roughly 3–7 hours of training, plus about 30 minutes of setup and export. That's an estimate,
because nobody has run this on a Mac yet. Step 3 gets you the real number within minutes. Your Mac can sit
unattended during training, but it needs to stay awake and plugged in.

**Why we're doing this:** RQ2 asks whether a cheap local model, fine-tuned on real negotiations, can compete
with a prompted hosted model. Agent C is the fine-tuned model with Agent A's prompt, and Agent D is the
fine-tuned model in Agent B's loop. No agent code is needed: both load from config once the model is in Ollama.

## Before you start

You need `uv`, Ollama, Xcode command line tools, `cmake` (`brew install cmake`) and about 25 GB of free disk.

```bash
git fetch && git switch role2 && git pull
uv sync
ollama pull qwen2.5:3b
mkdir -p runs/finetune
```

## 1. Build the training data

```bash
uv run python -m finetune.prepare_data
shasum -a 256 data/craigslist/sft/train.jsonl data/craigslist/sft/valid.jsonl
```

This downloads CraigslistBargain and converts it. Everything lands in git-ignored folders. You should get
3,141 training and 337 held-out examples, with these checksums:

```text
6bb28826ebb4e55d269ef6f9b5fef5c395bd820142cc2209d8371c8bc1e59ad7  train.jsonl
40752a2545f89b2a3aa2c760f7a3111efb0d2f94a3289c0aa29f996d9ec18ea2  valid.jsonl
```

If they differ, tell ZY before training, because the two machines would be training on different data.

## 2. Download the base models

These are pinned to exact versions, so the run can be reproduced:

```bash
uv run --no-project --with huggingface_hub python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("mlx-community/Qwen2.5-3B-Instruct-4bit", revision="4f83f8f146fdf28b512a06562b671d7af4fab457", local_dir="runs/finetune/base-4bit")
snapshot_download("mlx-community/Qwen2.5-3B-Instruct-bf16", revision="bec1ad658e380044bbfe06d4a2268801d3379a11", local_dir="runs/finetune/base-bf16")
EOF
```

## 3. Train

Settings are in `finetune/mlx_lora.yaml`. Close heavy apps first, then run:

```bash
caffeinate -i uv run --with mlx-lm==0.32.2 mlx_lm.lora --config finetune/mlx_lora.yaml 2>&1 | tee runs/finetune/train.log
```

**Check the speed after the first few rows.** The `tok/s` column gives the real time. The full run is about
5.0M tokens:

| `tok/s` | Training time |
|---|---|
| 500 | 2.8 hours |
| 300 | 4.7 hours |
| 200 | 7 hours |
| Under 100 | Over 14 hours: stop and use the Colab route below |

**If it runs out of memory** or the Mac starts swapping badly, stop it and set `batch_size: 2`,
`grad_accumulation_steps: 4` and `iters: 3141` in the YAML, plus `arguments: [2.0e-4, 3141, 1.0e-6]`. This is
the same training, just slower.

**If it gets interrupted,** checkpoints are saved every 100 iterations. To resume from, say, iteration 800,
add this, and use `tee -a` so the log isn't overwritten:

```bash
--resume-adapter-file runs/finetune/adapters/0000800_adapters.safetensors --iters 770
```

The learning-rate schedule restarts when you resume, so tell ZY if this happens. It needs noting in the report.

**What healthy looks like:** training and `val` loss both fall, then level off. If `val` loss climbs steadily
while training loss keeps falling, it's overfitting: you can stop early, because step 4 picks the best
checkpoint anyway. If the loss is `nan`, or doesn't move at all after a few hundred iterations, stop and tell ZY.

## 4. Pick the best checkpoint

```bash
uv run --group analysis python -m finetune.plot_loss runs/finetune/train.log
cp runs/finetune/adapters/<the file it names> runs/finetune/adapters/adapters.safetensors
```

This writes `loss_curve.png` and `training_log.json`, and names the checkpoint with the lowest held-out loss.
MLX's last checkpoint isn't necessarily its best one.

## 5. Export for Ollama

This fuses the adapter into the full-precision weights, then converts them to Q4_K_M, the same format as stock
`qwen2.5:3b`. Fusing onto the full-precision base avoids building on the rounded 4-bit weights.

```bash
uv run --with mlx-lm==0.32.2 mlx_lm.fuse --model runs/finetune/base-bf16 --adapter-path runs/finetune/adapters --save-path runs/finetune/fused

git clone --branch b6000 --depth 1 https://github.com/ggml-org/llama.cpp runs/finetune/llama.cpp
uv run --no-project --with torch==2.6.0 --with transformers==4.51.3 --with numpy --with sentencepiece --with protobuf --with safetensors \
  python runs/finetune/llama.cpp/convert_hf_to_gguf.py runs/finetune/fused --outtype f16 --outfile runs/finetune/fused-f16.gguf

cmake -S runs/finetune/llama.cpp -B runs/finetune/llama.cpp/build -DLLAMA_CURL=OFF
cmake --build runs/finetune/llama.cpp/build --target llama-quantize -j
runs/finetune/llama.cpp/build/bin/llama-quantize runs/finetune/fused-f16.gguf runs/finetune/qwen2.5-3b-haggle-q4_k_m.gguf q4_k_m
```

Expect a GGUF of about 1.9 GB. You can delete `fused/` and `fused-f16.gguf` afterwards.

Ollama can't do this step itself: version 0.40.14 refused to import Qwen from safetensors.

## 6. Serve and check it

```bash
cp finetune/Modelfile runs/finetune/Modelfile
(cd runs/finetune && ollama create qwen2.5-3b-haggle -f Modelfile)
uv run python -m finetune.compare_models qwen2.5:3b qwen2.5-3b-haggle --out runs/finetune/report.json
```

This compares both models on 100 held-out seller turns, without forcing the JSON format:

| In `report.json` | What we want from the tuned model |
|---|---|
| `valid_json_rate` | High, ideally near 1.0 |
| `same_action_as_human` | Higher than stock |
| `offers_below_floor` | Close to 0. Floors were made up to fit the data, so a high rate is a real warning sign |
| `median_price_error` | Lower than stock |
| `tokens_per_second` | Similar to stock. A big drop suggests an export problem |

A tuned model that does no better than stock is still a result, so don't rerun with new settings to chase a
better number. Tell ZY, as any change has to be reported.

## 7. Send back to ZY

Put these on Google Drive, as they're too big for git and `runs/` is ignored anyway:

- `runs/finetune/qwen2.5-3b-haggle-q4_k_m.gguf`
- `runs/finetune/adapters/adapters.safetensors` and `adapter_config.json`
- `runs/finetune/train.log`, `training_log.json`, `loss_curve.png`, `report.json`
- A short note: Mac memory (GB), total training time, typical `tok/s`, and any change to the settings or any resume

Don't post the weights or the dataset publicly: see Licences below.

## Fallback: Colab

Use this if MLX fails or is far too slow. It needs only a browser, so the Mac's hardware doesn't matter.

1. Open `finetune/qlora_colab.ipynb` in Colab and pick a GPU (Runtime > Change runtime type).
2. Run all. It mounts Google Drive and asks you to upload `train.jsonl` and `valid.jsonl`.
3. If the session drops, run all again. It resumes from the last checkpoint on Drive. `report.json` only counts the
   time of the last session, so add up the sessions in your note to ZY.
4. Download the GGUF from `MyDrive/haggle_finetune`, then do step 6. Move it to `runs/finetune/` first.

## What has and hasn't been tested

| Part | Status |
|---|---|
| `prepare_data.py`, `compare_models.py`, `plot_loss.py` helpers | Unit tests pass. `plot_loss` was only tested on a log written by hand to match MLX's source code, so **check it against your real log** |
| Prompts served by the Modelfile | Token counts matched the training prompts on 4 examples, using stock weights |
| A live game through the Modelfile | Ran cleanly on stock weights |
| Notebook training, resuming, scoring, merging | Ran for a few steps on an RTX 2060, on Windows. This tested the code, not model quality |
| **Everything on MLX: training, fusing, `mlx_lora.yaml`** | **Never run.** MLX doesn't run on the Windows machine. Options were checked against the mlx-lm 0.32.2 source |
| **`llama.cpp` GGUF conversion and quantizing** | **Never run** on either route |
| Colab itself | Never run. Its setup, Drive and export cells only run there |

If something in the untested parts fails, it's probably a problem with these instructions rather than with
your setup, so tell ZY.

## Why the data looks the way it does

- **Which dialogues:** of 5,247 training dialogues, 671 were kept: 580 good deals and 91 refusals of offers below
  the floor. `data/craigslist/sft/funnel.json` shows why the rest were dropped.
- **Floors are synthetic:** the dataset has none, so one is drawn like `make_scenario` does, then adjusted to fit
  what the seller did.
- **`wait_hours` is always 0:** the dataset has no usable timing, so the model won't learn pacing.
- **No condition facts:** the prompt's facts section is a neutral placeholder, so this data can't teach honesty.
- **The published test split isn't used:** it has no price annotations. The held-out set is the dev split.

## Licences

- **Qwen2.5-3B** is under the Qwen Research Licence, so research use only.
- **CraigslistBargain** lists its licence as unknown on Hugging Face. Cite He et al. (2018), and don't redistribute it.
