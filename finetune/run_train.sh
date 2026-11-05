#!/bin/bash
set -e

# Disable MLX distributed MPI loading (prevents conflict with Anaconda's MPICH)
export MLX_MPI_LIBNAME=nonexistent.dylib

echo "Starting MLX LoRA training..."
echo "Mac will be kept awake using caffeinate."

caffeinate -i uv run --with mlx-lm==0.32.2 python -m mlx_lm lora --config finetune/mlx_lora.yaml 2>&1 | tee runs/finetune/train.log
