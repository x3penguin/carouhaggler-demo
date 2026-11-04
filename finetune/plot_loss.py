"""Loss curve and best checkpoint from an MLX training log (the notebook makes its own curve).

    uv run --group analysis python -m finetune.plot_loss runs/finetune/train.log

MLX keeps every saved checkpoint rather than the best one, so this also names the checkpoint with
the lowest held-out loss: that's the one to fuse, not the last one.
"""

import argparse
import json
import re
from pathlib import Path

_TRAIN = re.compile(r"^\s*(\d+)\s+(\d+\.\d+)\s*[▼▲]")
_VAL = re.compile(r"^\s*(\d+)\s+val\s+(\d+\.\d+)")


def parse_log(text: str) -> dict:
    """Pull (iteration, loss) pairs from mlx_lm.lora's printed table, ignoring everything else."""
    train, val = [], []
    for line in text.splitlines():
        if match := _TRAIN.match(line):
            train.append((int(match[1]), float(match[2])))
        elif match := _VAL.match(line):
            val.append((int(match[1]), float(match[2])))
    return {"train": train, "val": val}


def best_checkpoint(val: list[tuple[int, float]], save_every: int = 100) -> tuple[int, float] | None:
    """The lowest held-out loss at an iteration that has a saved checkpoint."""
    saved = [(it, loss) for it, loss in val if it % save_every == 0]
    return min(saved, key=lambda point: point[1]) if saved else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path)
    args = parser.parse_args()

    curves = parse_log(args.log.read_text(encoding="utf-8", errors="replace"))
    if not curves["train"]:
        raise SystemExit(f"No training rows found in {args.log}. Is it the output of mlx_lm.lora?")
    best = best_checkpoint(curves["val"])
    out_dir = args.log.parent
    (out_dir / "training_log.json").write_text(json.dumps({**curves, "best": best}, indent=1), encoding="utf-8")

    import matplotlib.pyplot as plt  # only this step needs the analysis group

    plt.plot(*zip(*curves["train"]), label="training loss")
    if curves["val"]:
        plt.plot(*zip(*curves["val"]), marker="o", label="held-out loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("QLoRA fine-tune of the seller")
    plt.legend()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    if best:
        print(f"Best held-out loss {best[1]:.3f} at iteration {best[0]}: use {best[0]:07d}_adapters.safetensors")


if __name__ == "__main__":
    main()
