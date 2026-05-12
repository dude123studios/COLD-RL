"""
Entry point for COLD-RL training.

Spec §1.6 defaults:
  I=16 rollouts, batch=256 (16 problems × 16 rollouts)
  LR: 1e-5 for 7B; 1e-4 for ≤3B
  β=0.01, ε=0.2, λ=0.5 (post-ramp), 8 epochs
  Data: NuminaMath (short-CoT) or OpenThoughts-3 (long-CoT)

Usage:
    # E1 — 7B primary
    python -m rl.run_rl --model Qwen/Qwen2.5-7B --lr 1e-5 \
        --output_dir /data/user_data/shivansg/cold_rl_runs/e1_7b

    # E1 — 0.5B
    python -m rl.run_rl --model Qwen/Qwen2.5-0.5B --lr 1e-4 \
        --output_dir /data/user_data/shivansg/cold_rl_runs/e1_0p5b

    # GRPO baseline (λ=0, epoch 1 only has λ=0 anyway but set n_epochs too)
    python -m rl.run_rl --model Qwen/Qwen2.5-7B --grpo_baseline \
        --output_dir /data/user_data/shivansg/cold_rl_runs/grpo_baseline_7b
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.diversity_grpo import DiversityGRPOConfig, find_latest_checkpoint, train


def parse_args() -> DiversityGRPOConfig:
    p = argparse.ArgumentParser(description="COLD-RL training")

    # Model
    p.add_argument("--model", default="Qwen/Qwen2.5-7B", dest="model_name",
                   help="HuggingFace model ID. Use BASE models for RL training.")

    # Data (spec §1.6)
    p.add_argument("--dataset", default="numinamath",
                   help="numinamath | openthoughts3 | math | deepscaler | gsm8k | <path>")
    p.add_argument("--dataset_path", default=None)
    p.add_argument("--benchmark", default="math",
                   help="Benchmark for answer grading during training spot-checks")

    # Rollouts (spec §1.2: I=16 fixed)
    p.add_argument("--n_rollouts", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_new_tokens", type=int, default=8192)

    # Embedding for diversity reward (spec: Qwen3-8B; local for practical cluster use)
    p.add_argument("--embed_model", default="local",
                   choices=["local", "qwen3", "openrouter"],
                   help="local=e5-small-v2 (fast); qwen3=Qwen3-Embedding (spec); "
                        "openrouter=API (requires OPENROUTER_API_KEY)")

    # GRPO hyperparams (spec §1.6)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="1e-5 for 7B; 1e-4 for ≤3B (spec §1.6)")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--kl_beta", type=float, default=0.01)

    # Training schedule (spec §1.6: 8 epochs, batch=256)
    p.add_argument("--n_epochs", type=int, default=8)
    p.add_argument("--n_problems_per_step", type=int, default=16,
                   help="16 problems × 16 rollouts = 256 completions per step")
    p.add_argument("--mini_batch", type=int, default=8)
    p.add_argument("--ref_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # GRPO-only baseline convenience flag (forces λ=0 for all epochs)
    p.add_argument("--grpo_baseline", action="store_true",
                   help="Run pure GRPO (λ=0). Overrides epoch curriculum to keep λ=0.")

    # LoRA
    p.add_argument("--use_lora", action="store_true", default=True)
    p.add_argument("--no_lora", dest="use_lora", action="store_false")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)

    # Output
    p.add_argument("--output_dir", default="results/rl_runs/cold_rl")
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--resume_from", default=None)

    # Hardware
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    d = vars(args)

    # --grpo_baseline: monkey-patch _lambda_for_epoch to always return 0
    grpo_baseline = d.pop("grpo_baseline", False)
    if grpo_baseline:
        import rl.diversity_grpo as _dg
        _dg._lambda_for_epoch = lambda epoch: 0.0

    return DiversityGRPOConfig(**d)


if __name__ == "__main__":
    cfg = parse_args()

    # Auto-resume: find latest checkpoint in output_dir if not explicitly given
    if cfg.resume_from is None:
        latest = find_latest_checkpoint(Path(cfg.output_dir))
        if latest is not None:
            logging.getLogger(__name__).info(
                f"[auto-resume] Found {latest.name} — resuming"
            )
            cfg.resume_from = str(latest)
        else:
            logging.getLogger(__name__).info("[auto-resume] Starting fresh")

    print("=" * 60)
    print("COLD-RL Training")
    print("=" * 60)
    print(f"  Model:       {cfg.model_name}")
    print(f"  Dataset:     {cfg.dataset}")
    print(f"  Rollouts:    {cfg.n_rollouts}  (I=16 per spec)")
    print(f"  Epochs:      {cfg.n_epochs}  (λ: 0→0.17→0.33→0.50)")
    print(f"  Embed:       {cfg.embed_model}")
    print(f"  LR:          {cfg.lr}")
    print(f"  Batch:       {cfg.n_problems_per_step} × {cfg.n_rollouts} = "
          f"{cfg.n_problems_per_step * cfg.n_rollouts} completions/step")
    print(f"  Output:      {cfg.output_dir}")
    if cfg.resume_from:
        print(f"  Resume:      {cfg.resume_from}")
    print("=" * 60)

    train(cfg)
