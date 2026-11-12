# DPO (preference training) on the fine-tuned seller

The SFT model (`qwen2.5-3b-haggle`, called **v1**; see [`../README.md`](../README.md)) sold below its secret floor in
38% of games and told major lies about the item in 37–50%. We tried to fix this with **DPO**: instead of one right
answer, the model sees the same chat with a *preferred* and a *rejected* reply and learns to prefer the first. Three
versions were trained, each fixing the previous one's failure. Every version trains a new LoRA adapter on top of the
SFT model, so they are siblings, not a chain.

| Folder | Pairs | Built by | What changed | What happened |
|---|---|---|---|---|
| [`v2/`](v2) | 3,700 train (no held-out split) | `finetune/make_dpo_pairs.py` | Three rules: don't accept below the floor, don't reveal it, don't invent facts | Learned "never accept": Agent A's deal rate fell from 80% to 23%, and it refused even a buyer offering its whole budget |
| [`v3/`](v3) | 3,043 train, 351 valid | `finetune/build_dpo_pairs.py --seed 42` | Added pairs where accepting a fair offer is right | Closed deals again (9 of 10 against a full-budget buyer) but gave in to pressure: lost 57 of 200 attack games |
| [`v4/`](v4) | 4,119 train, 454 valid | `finetune/build_dpo_pairs_v4.py --seed 42` | Every prompt twice, with the floor just above and just below the offer, so the floor is the only clue; buyer pressure and opening offers; an SFT loss term | Trained cleanly, but **failed the decision check**: it accepts about 70% of offers whether they are above or below the floor. DPO stopped here |

Each folder holds the `train.jsonl` / `valid.jsonl` pairs and the `dpo_colab.ipynb` notebook that trained that
version. All three pair files are rebuilt **byte for byte** by their builder (after step 1 of
[`../README.md`](../README.md), which creates the SFT data they read). The full design log, with the reasoning,
audit numbers and every check, is [`../DPO_V3.md`](../DPO_V3.md).

**Conclusion:** each version changed *how often* the model accepts, never *when*. None learned to compare the offer
with its private floor, so the floor rule has to live in code (the engine's floor guardrail), not in the model.

## Download the trained models

| Model | Ollama name | `configs/models.yaml` key | SHA-256 | Link |
|---|---|---|---|---|
| `qwen2.5-3b-haggle-dpo-q4_k_m.gguf` (v2) | `qwen2.5-3b-haggle-dpo` | `finetuned_dpo` | `281be40d0c225b7d85c09e5762e05de754f4ddc0364c0e3972073fa226950c9f` | [download](LINK_TO_DPO_V2_GGUF) |
| `qwen2.5-3b-haggle-dpo-v3-q4_k_m.gguf` | `qwen2.5-3b-haggle-dpo-v3` | `finetuned_dpo_v3` | `2c98a50508f8574ced40292f42b7aef4387d477e510683abb5a0cca7d1d8240a` | [download](LINK_TO_DPO_V3_GGUF) |
| `qwen2.5-3b-haggle-dpo-v4-q4_k_m.gguf` | `qwen2.5-3b-haggle-dpo-v4` | `finetuned_dpo_v4` | `1aa7a7c6b69fd3a3944e38b96b3565c1979460937faa7d6a3b26a639a044ec37` | [download](LINK_TO_DPO_V4_GGUF) |

Each GGUF is about 1.9 GB. Register one with Ollama by copying `finetune/Modelfile` next to it and pointing `FROM` at
the file, e.g. for v4:

```bash
cp finetune/Modelfile runs/finetune/Modelfile.dpo-v4
sed -i 's#^FROM .*#FROM ./qwen2.5-3b-haggle-dpo-v4-q4_k_m.gguf#' runs/finetune/Modelfile.dpo-v4   # macOS: sed -i ''
(cd runs/finetune && ollama create qwen2.5-3b-haggle-dpo-v4 -f Modelfile.dpo-v4)
```

## Retrain a version

1. **Build the pairs** (or use the committed ones). Needs the SFT data from step 1 of [`../README.md`](../README.md):

   ```bash
   uv run python finetune/make_dpo_pairs.py                     # v2 -> finetune/dpo/v2/
   uv run python -m finetune.build_dpo_pairs --seed 42          # v3 -> finetune/dpo/v3/
   uv run python -m finetune.build_dpo_pairs_v4 --seed 42       # v4 -> finetune/dpo/v4/
   ```

2. **Audit them** (free, no model calls). The audit has 11 checks, e.g. accept wins about as often as it loses,
   offer size alone can't predict the answer, no preferred reply states the floor, and no phrase is copied from
   `data/attacks.json`. v4 passes all of them. v2 fails 7 and v3 fails 2 (accept balance, and offer size predicting
   the answer), which is how we found the shortcuts they learned.

   ```bash
   uv run python -m finetune.audit_dpo_pairs finetune/dpo/v4/train.jsonl
   ```

3. **Train** with that version's notebook, `finetune/dpo/<version>/dpo_colab.ipynb`, on Colab (pick a GPU runtime) or
   any CUDA machine with Jupyter. Upload the SFT adapter `adapters.safetensors` (from the SFT training, or the link in
   [`../README.md`](../README.md)) and the version's `train.jsonl` (v4 also `valid.jsonl`), then Run all. The
   notebook converts the MLX adapter to PEFT, merges it into the base model, trains a new LoRA with TRL's
   `DPOTrainer`, then merges and exports a Q4_K_M GGUF with llama.cpp.

4. **Register** the GGUF with Ollama as above.

5. **Check it for free before any paid run:**

   ```bash
   uv run python -m finetune.check_decisions finetune/dpo/v4/valid.jsonl --models finetuned finetuned_dpo_v4   # accepts only above the floor?
   uv run python -m haggle.runner configs/closing_check.yaml    # still closes deals? (add the new model as a seller line)
   ```

6. **Run the experiments** (scripted attackers and a local seller cost nothing; the tournament pays for LLM buyers):

   ```bash
   uv run python -m haggle.runner configs/attacks_dpo.yaml       # v2, 400 games
   uv run python -m haggle.runner configs/tournament_dpo.yaml    # v2, 240 games, about $0.27 of buyer calls
   uv run python -m haggle.runner configs/attacks_dpo_v3.yaml    # v3, 400 games
   ```

The committed results are in `runs/attacks_dpo/`, `runs/tournament_dpo/`, `runs/attacks_dpo_v3/`,
`runs/closing_check/` and `finetune/check_decisions_results.json` (the decision check for base, SFT, v2, v3 and v4).

## Training settings

All versions: base `Qwen/Qwen2.5-3B-Instruct` at revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` with the SFT
adapter merged in; a new LoRA with rank 16, alpha 32, dropout 0.05 on q/k/v/o projections; `beta` 0.1; learning rate
5e-6; one epoch; effective batch 8.

| | v2 | v3 | v4 |
|---|---|---|---|
| Loss | sigmoid (standard DPO) | sigmoid (the SFT term it asked for was most likely skipped: current TRL has no `rpo_alpha`) | `["sigmoid", "sft"]`, equal weights |
| `max_length` | TRL default (1,024 on current TRL) | TRL default (cut 315 of 3,043 pairs at 1,024) | 1,536 (no pair cut) |
| Held-out eval | none | none | `valid.jsonl` every 100 steps |
| Library versions | latest at training time | latest at training time | TRL 1.17.1, transformers 5.22.2 (logged by the notebook) |

The v4 run: 515 steps, held-out reward accuracy about 0.87, eval loss 1.64 → 1.33. Training looked healthy, which is
why the decision check matters: ranking whole replies correctly is not the same as making the right move.

**Optional local route for v2:** `finetune/train_dpo_local.py` (run with `bash finetune/run_dpo.sh`) is an alternative
v2 training script that runs without Colab, using PyTorch's MPS backend on Apple-silicon Macs.
