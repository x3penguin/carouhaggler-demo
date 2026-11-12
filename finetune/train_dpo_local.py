#!/usr/bin/env python3
"""Local DPO fine-tuning on Apple Silicon (MPS).

Trains a LoRA DPO adapter directly on top of the pre-fused v1 model (runs/finetune/fused),
evaluates contrastive loss on finetune/dpo/v2/train.jsonl, merges weights, and exports GGUF for Ollama.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def main():
    repo_root = Path(__file__).resolve().parent.parent
    fused_v1_dir = repo_root / "runs" / "finetune" / "fused"
    dpo_data_path = repo_root / "finetune" / "dpo" / "v2" / "train.jsonl"
    out_dir = repo_root / "runs" / "finetune" / "dpo_v2_fused"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not fused_v1_dir.exists():
        sys.exit(f"Error: Fused v1 model not found at {fused_v1_dir}")
    if not dpo_data_path.exists():
        sys.exit(f"Error: DPO training pairs not found at {dpo_data_path}")

    # 1. Device check
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        print("Warning: MPS not available, falling back to CPU")

    # 2. Load tokenizer and pre-fused v1 model
    print(f"\n[1/5] Loading pre-fused v1 model from {fused_v1_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(fused_v1_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(fused_v1_dir),
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.config.use_cache = False

    # 3. LoRA config for DPO adapter
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Format DPO dataset
    print(f"\n[2/5] Formatting DPO dataset from {dpo_data_path}...")
    raw_data = []
    with open(dpo_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))

    formatted_rows = []
    for item in raw_data:
        prompt_str = tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted_rows.append({
            "prompt": prompt_str,
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        })

    dataset = Dataset.from_list(formatted_rows)
    print(f"Loaded {len(dataset)} preference pairs.")

    # 5. DPO Training with live tqdm progress bar
    print("\n[3/5] Starting DPO training on MPS...")
    import inspect

    dpo_config_params = inspect.signature(DPOConfig.__init__).parameters
    config_kwargs = {
        "output_dir": str(out_dir / "checkpoints"),
        "learning_rate": 5e-6,
        "beta": 0.1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 1,
        "bf16": False,
        "fp16": False,
        "logging_steps": 10,
        "save_strategy": "no",
        "remove_unused_columns": False,
        "report_to": "none",
        "disable_tqdm": False,
    }
    if "max_length" in dpo_config_params:
        config_kwargs["max_length"] = 1536
    if "max_prompt_length" in dpo_config_params:
        config_kwargs["max_prompt_length"] = 1024

    training_args = DPOConfig(**config_kwargs)

    trainer_params = inspect.signature(DPOTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "peft_config": peft_config,
        "train_dataset": dataset,
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    if "max_length" in trainer_params:
        trainer_kwargs["max_length"] = 1536
    if "max_prompt_length" in trainer_params:
        trainer_kwargs["max_prompt_length"] = 1024

    trainer = DPOTrainer(**trainer_kwargs)

    trainer.train()
    print("DPO optimization completed!")

    # 6. Merge DPO adapter into final v2 model
    print(f"\n[4/5] Merging DPO weights into {out_dir}...")
    final_model = trainer.model.merge_and_unload()
    final_model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print("Fused v2 model saved.")

    # 7. Quantize to GGUF using existing local llama.cpp
    print("\n[5/5] Exporting to Ollama GGUF (Q4_K_M)...")
    llama_dir = repo_root / "runs" / "finetune" / "llama.cpp"
    convert_py = llama_dir / "convert_hf_to_gguf.py"
    quantize_bin = llama_dir / "build" / "bin" / "llama-quantize"

    f16_gguf = out_dir / "fused-f16.gguf"
    q4_gguf = repo_root / "finetune" / "qwen2.5-3b-haggle-dpo-q4_k_m.gguf"

    cmd_convert = [
        sys.executable, str(convert_py),
        str(out_dir),
        "--outtype", "f16",
        "--outfile", str(f16_gguf),
    ]
    subprocess.run(cmd_convert, check=True)

    cmd_quant = [
        str(quantize_bin),
        str(f16_gguf),
        str(q4_gguf),
        "q4_k_m",
    ]
    subprocess.run(cmd_quant, check=True)

    if f16_gguf.exists():
        f16_gguf.unlink()

    print(f"\nSUCCESS! Quantized v2 model exported to:\n  {q4_gguf}")
    print("\nTo load into Ollama, run:")
    print(f"  ollama create qwen2.5-3b-haggle-v2 -f finetune/Modelfile_dpo")


if __name__ == "__main__":
    main()
