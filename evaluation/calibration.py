"""
COLD-RL Phase 1 — Calibration (spec §2.1)

Computes P̂(n, i) — the estimated probability that at least one of n draws
from attempt-index i solves a random problem — and the marginal gain
Δ(n, i) = P̂(n+1, i) − P̂(n, i).

These tables drive the water-filling allocator (see water_filling.py).

Algorithm
---------
For each held-out problem x and each attempt index i ∈ {1,...,I}:
  1. Generate R=64 independent draws at T=1.0.
  2. Count incorrect draws: F_i(x) = number of wrong answers.
  3. Estimate pass@n using the unbiased order-statistics estimator:
       P̂(n, i) = (1/N) Σ_x [ 1 − C(F_i(x), n) / C(R, n) ]
     where C(a, b) = 0 if a < b, else comb(a, b).
     (Uses log-space arithmetic to handle large C values accurately.)

Usage
-----
    python -m evaluation.calibration \
        --model Qwen/Qwen2.5-7B \
        --lora-path /data/.../step_08000 \
        --n-problems 500 \
        --R 64 \
        --out results/calibration/e1_7b.json
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.diversity_grpo import ATTEMPT_PREFIX, MATH_INSTRUCTION, I_ROLLOUTS

logger = logging.getLogger(__name__)

R_DEFAULT = 64
I_DEFAULT = I_ROLLOUTS   # 16


# ---------------------------------------------------------------------------
# Combinatorics helpers (log-space for safety)
# ---------------------------------------------------------------------------

def _log_comb(n: int, k: int) -> float:
    """log C(n, k) using lgamma; returns -inf if n < k."""
    if k < 0 or k > n:
        return -math.inf
    if k == 0 or k == n:
        return 0.0
    return (math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1))


def _pass_at_n(F: int, R: int, n: int) -> float:
    """
    Unbiased pass@n for a single problem.
    F = number of failures out of R draws.
    p@n = 1 − C(F, n) / C(R, n)
    """
    if n > R:
        n = R
    if n <= 0:
        return 0.0
    log_num = _log_comb(F, n)
    log_den = _log_comb(R, n)
    if log_den == -math.inf:
        return 0.0
    if log_num == -math.inf:
        return 1.0
    return float(1.0 - math.exp(log_num - log_den))


# ---------------------------------------------------------------------------
# P̂(n, i) table computation
# ---------------------------------------------------------------------------

def compute_phat(
    failure_counts: np.ndarray,   # shape (N_problems, I)
    R: int,
    n_max: int,
) -> np.ndarray:
    """
    Compute P̂(n, i) for n ∈ {0,...,n_max} and i ∈ {0,...,I-1}.

    Args
    ----
    failure_counts : (N, I) int array; failure_counts[x, i] = F_i(x)
    R              : total draws per (problem, index)
    n_max          : maximum n to compute

    Returns
    -------
    phat : (n_max+1, I) float array; phat[n, i] = P̂(n, i)
    """
    N, I = failure_counts.shape
    phat = np.zeros((n_max + 1, I), dtype=np.float64)

    for i in range(I):
        for n in range(n_max + 1):
            vals = [_pass_at_n(int(F), R, n) for F in failure_counts[:, i]]
            phat[n, i] = float(np.mean(vals))

    return phat


def compute_delta(phat: np.ndarray) -> np.ndarray:
    """
    Δ(n, i) = P̂(n+1, i) − P̂(n, i).
    Returns array of shape (n_max, I).
    """
    return phat[1:] - phat[:-1]   # (n_max, I)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _build_prompts(problems: list[dict], I: int) -> dict[str, str]:
    """Build {prompt_id: raw_prompt_text} for all (problem, attempt_index) pairs."""
    from evaluation.vllm_generate import _format_chat

    prompts: dict[str, str] = {}
    for p in problems:
        for i in range(1, I + 1):
            key = f"{p['id']}__i{i}"
            if i == 1:
                content = f"{p['problem']}\n\n{MATH_INSTRUCTION}"
            else:
                prefix = ATTEMPT_PREFIX.format(i=i)
                content = f"{prefix}\n\n{p['problem']}\n\n{MATH_INSTRUCTION}"
            prompts[key] = _format_chat(content)
    return prompts


def _score_response(response: str, gold: str, benchmark: str = "math") -> bool:
    from rl.diversity_reward import is_correct
    return is_correct(response, gold, benchmark)


# ---------------------------------------------------------------------------
# Main calibration routine
# ---------------------------------------------------------------------------

def run_calibration(
    model: str,
    lora_path: Optional[str],
    problems: list[dict],
    benchmark: str = "math",
    R: int = R_DEFAULT,
    I: int = I_DEFAULT,
    n_max: int = 64,
    out_path: Optional[str] = None,
) -> dict:
    """
    Run Phase 1 calibration.  Generates R draws per (problem, index) pair,
    counts failures, and computes P̂(n,i) and Δ(n,i) tables.

    Returns a dict with keys:
      phat       : (n_max+1) × I list-of-lists
      delta      : n_max × I list-of-lists
      failure_counts : N × I list-of-lists
      meta       : run metadata
    """
    import subprocess, tempfile

    N = len(problems)
    logger.info(f"[calib] {N} problems × {I} indices × {R} draws = {N*I*R} total completions")

    # Build all prompts
    prompts = _build_prompts(problems, I)   # N*I prompt strings

    # Run vLLM worker: n=R per prompt, temperature=1.0
    with tempfile.TemporaryDirectory() as tmpdir:
        prompts_file = os.path.join(tmpdir, "prompts.jsonl")
        out_file     = os.path.join(tmpdir, "rollouts.jsonl")

        with open(prompts_file, "w") as f:
            for pid, prompt in prompts.items():
                f.write(json.dumps({"id": pid, "prompt": prompt}) + "\n")

        cmd = [
            sys.executable, "-m", "evaluation.vllm_worker",
            "--model", model,
            "--prompts-file", prompts_file,
            "--out-file", out_file,
            "--n", str(R),
            "--temperature", "1.0",
            "--max-tokens", "8192",
        ]
        if lora_path:
            cmd += ["--lora-path", lora_path]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent.parent) + ":" + env.get("PYTHONPATH", "")
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"vllm_worker exited {result.returncode}")

        rollouts: dict[str, list[str]] = {}
        with open(out_file) as f:
            for line in f:
                row = json.loads(line)
                rollouts[row["id"]] = row["rollouts"]

    # Count failures F_i(x)
    failure_counts = np.zeros((N, I), dtype=np.int32)
    gold_map = {p["id"]: p["answer"] for p in problems}

    for pi, prob in enumerate(problems):
        gold = gold_map[prob["id"]]
        for i in range(1, I + 1):
            key = f"{prob['id']}__i{i}"
            draws = rollouts.get(key, [])
            n_correct = sum(_score_response(r, gold, benchmark) for r in draws)
            failure_counts[pi, i - 1] = R - n_correct

    # Compute P̂ and Δ tables
    phat  = compute_phat(failure_counts, R=R, n_max=n_max)
    delta = compute_delta(phat)

    result = {
        "meta": {
            "model": model,
            "lora_path": lora_path,
            "benchmark": benchmark,
            "n_problems": N,
            "R": R,
            "I": I,
            "n_max": n_max,
        },
        "phat":           phat.tolist(),    # (n_max+1, I)
        "delta":          delta.tolist(),   # (n_max, I)
        "failure_counts": failure_counts.tolist(),  # (N, I)
    }

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, indent=2))
        logger.info(f"[calib] Saved → {out_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="COLD-RL Phase 1 calibration")
    ap.add_argument("--model",       required=True)
    ap.add_argument("--lora-path",   default=None)
    ap.add_argument("--benchmark",   default="math")
    ap.add_argument("--n-problems",  type=int, default=500,
                    help="Held-out calibration problems (spec: 500)")
    ap.add_argument("--R",           type=int, default=R_DEFAULT,
                    help="Draws per (problem, index) pair (spec: 64)")
    ap.add_argument("--I",           type=int, default=I_DEFAULT,
                    help="Number of attempt indices (spec: 16)")
    ap.add_argument("--n-max",       type=int, default=64,
                    help="Maximum n to compute in P̂(n,i)")
    ap.add_argument("--dataset",     default="math500")
    ap.add_argument("--out",         required=True)
    args = ap.parse_args()

    from evaluation.vllm_generate import load_benchmark
    problems = load_benchmark(args.dataset)
    import random; random.shuffle(problems)
    problems = problems[:args.n_problems]
    logger.info(f"[calib] Using {len(problems)} problems from {args.dataset}")

    run_calibration(
        model=args.model,
        lora_path=args.lora_path,
        problems=problems,
        benchmark=args.benchmark,
        R=args.R,
        I=args.I,
        n_max=args.n_max,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
