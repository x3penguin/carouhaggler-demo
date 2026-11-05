from huggingface_hub import snapshot_download

print("Downloading Qwen2.5-3B-Instruct-4bit...")
snapshot_download(
    "mlx-community/Qwen2.5-3B-Instruct-4bit",
    revision="4f83f8f146fdf28b512a06562b671d7af4fab457",
    local_dir="runs/finetune/base-4bit",
)

print("Downloading Qwen2.5-3B-Instruct-bf16...")
snapshot_download(
    "mlx-community/Qwen2.5-3B-Instruct-bf16",
    revision="bec1ad658e380044bbfe06d4a2268801d3379a11",
    local_dir="runs/finetune/base-bf16",
)

print("Base weights downloaded successfully.")
