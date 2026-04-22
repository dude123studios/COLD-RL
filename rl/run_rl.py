"""
Entry point for Diversity-GRPO training.

Thesis: Controllable Diversity
  Standard RL (GRPO) + correctness reward → diversity collapse: the model
  always produces the same solution. We want diversity ON DEMAND — when the
  user asks for "approach #X", the model produces a genuinely different method.

  Our reward teaches the model two things simultaneously:
    1. Correctness: solve the problem right
    2. Uniqueness of method: the approach you use must be distinct from all
       other methods used in the same rollout group

  At inference time, diversity is CONTROLLABLE:
    - Use standard prompt → model produces best answer (no diversity cost)
    - Use "approach #1 of 8" prompts → model produces 8 diverse correct solutions

  This is stronger than DARLING, which trains a ALWAYS-diverse model.
  Our method trains an ON-DEMAND-diverse model.

Supported datasets:
  - gsm8k       (7.5K training problems, local or HuggingFace)
  - math        (7.5K MATH training set, HuggingFace lighteval/MATH)
  - deepscaler  (10K competition math, HuggingFace agentica-org/DeepScaleR-Preview-Dataset)
                 — THIS is what DARLING trains on (Qwen3-4B-Base, 64 GPUs)
  - aime        (AI-MO AIME validation set)
  - Any .jsonl with 'problem'+'answer' fields via --dataset_path

Recommended for best results (hardest problems → most room for diverse methods):
  deepscaler > math > gsm8k

DARLING hyperparameters (from arXiv:2509.02534):
  --model Qwen/Qwen3-4B     (DARLING uses Qwen3-4B-Base)
  --n_rollouts 8
  --lr 1e-6
  --n_problems_per_step 32  (32 × 8 = 256 global batch)
  --max_new_tokens 8192
  --clip_eps 0.2
  --kl_beta 0.001
  --temperature 0.8

Usage:
    # Primary experiment (OpenRouter embeddings)
    OPENROUTER_API_KEY=... python -m rl.run_rl \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset deepscaler \
        --lambda_div 0.5 \
        --embed_model openrouter \
        --gpu_id 2 \
        --output_dir results/rl_runs/div_grpo_7b_deepscaler

    # Free ablation (local embeddings, no API cost)
    python -m rl.run_rl --embed_model local --lambda_div 0.5 ...

    # GRPO-only baseline (lambda=0, for comparison)
    python -m rl.run_rl --embed_model local --lambda_div 0.0 ...
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

from rl.diversity_grpo import DiversityGRPOConfig, train


def parse_args() -> DiversityGRPOConfig:
    p = argparse.ArgumentParser(description="Diversity-GRPO training")

    # Model
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                   dest="model_name", help="HuggingFace model ID")
    # Data
    p.add_argument("--dataset", default="gsm8k",
                   help="Dataset name (gsm8k | math | ...)")
    p.add_argument("--dataset_path", default=None,
                   help="Override path to .jsonl file with 'problem'/'answer' fields")
    p.add_argument("--benchmark", default="gsm8k",
                   help="Benchmark name for answer grading")

    # Rollouts (DARLING defaults)
    p.add_argument("--n_rollouts", type=int, default=8,
                   help="Number of diverse rollouts per problem (= N methods)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_new_tokens", type=int, default=8192)

    # Reward
    p.add_argument("--lambda_div", type=float, default=0.5,
                   help="Weight for diversity reward (added to correctness reward)")
    p.add_argument("--embed_model", default="openrouter",
                   choices=["openrouter", "local"],
                   help="Embedding model: openrouter=Qwen3-Embed-8B, local=MiniLM")
    p.add_argument("--use_xml_steps", action="store_true",
                   help="Extract steps via XML parsing (requires model to output <step> tags)")

    # GRPO hyperparams (DARLING)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--kl_beta", type=float, default=0.001)

    # Training schedule
    p.add_argument("--n_problems_per_step", type=int, default=32,
                   help="Problems per gradient step (×N rollouts = global batch size)")
    p.add_argument("--mini_batch", type=int, default=8,
                   help="Completions per backward pass (controls activation memory). "
                        "Use 8 on 48GB GPUs when vLLM shares the device; raise on 80GB if stable.")
    p.add_argument("--ref_batch_size", type=int, default=2,
                   help="Sequences per batched reference log-prob forward pass (no grad). "
                        "Small values (2–4) avoid lm_head logits OOM on 48GB cards.")
    p.add_argument("--total_steps", type=int, default=1000)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # LoRA
    p.add_argument("--use_lora", action="store_true", default=True)
    p.add_argument("--no_lora", dest="use_lora", action="store_false")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)

    # Output
    p.add_argument("--output_dir", default="results/rl_runs/div_grpo")
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--resume_from", default=None,
                   help="Path to a checkpoint dir (step_XXXXX/) with LoRA adapter + metrics.json")

    # Hardware
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    return DiversityGRPOConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()

    print("=" * 60)
    print("Diversity-GRPO Training")
    print("=" * 60)
    print(f"  Model:         {cfg.model_name}")
    print(f"  Dataset:       {cfg.dataset}")
    print(f"  N rollouts:    {cfg.n_rollouts}  (method prompts 1..{cfg.n_rollouts})")
    print(f"  lambda_div:    {cfg.lambda_div}")
    print(f"  embed_model:   {cfg.embed_model}")
    print(f"  LR:            {cfg.lr}")
    print(f"  Steps:         {cfg.total_steps}")
    print(f"  Global batch:  {cfg.n_problems_per_step} probs × {cfg.n_rollouts} rollouts = "
          f"{cfg.n_problems_per_step * cfg.n_rollouts} completions")
    print(f"  Output:        {cfg.output_dir}")
    if cfg.resume_from:
        print(f"  Resume:        {cfg.resume_from}")
    print("=" * 60)

    train(cfg)
