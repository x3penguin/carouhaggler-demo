#!/bin/bash
set -e

# Disable MLX distributed MPI loading (prevents conflict with Anaconda's MPICH)
export MLX_MPI_LIBNAME=nonexistent.dylib

echo "Starting local DPO alignment on Apple Silicon (MPS)..."
echo "Keeping Mac awake using caffeinate."

caffeinate -i uv run --with "torch,transformers,peft,trl,datasets,accelerate,gguf" python finetune/train_dpo_local.py
