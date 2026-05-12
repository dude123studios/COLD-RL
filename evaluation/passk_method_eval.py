"""
Pass@k evaluation using COLD-RL method prompts.

Two modes run automatically per checkpoint:
  METHOD: k prompts per problem (approach #1..#k), n=1 each — tests the COLD mechanism
  STANDARD: 1 prompt per problem, n=max_k completions at T=0.8 — temperature baseline

Outputs a JSON with pass@k for k in {1,2,4,8,16,32} for both modes.

Usage:
    python -m evaluation.passk_method_eval \
        --model Qwen/Qwen2.5-7B-Instruct \
        [--lora-path results/rl_runs/cold_lambda05_7b/step_00700] \
        --benchmarks math500 aime24 \
        --max-k 32 \
        --experiment-id cold_lambda05_7b \
        --out results/passk/cold_lambda05_7b.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Spec §1.1: must match training prefix exactly.
# i=1 → no prefix; i>1 → "[Attempt #i — use a new method]"
from rl.diversity_grpo import ATTEMPT_PREFIX, MATH_INSTRUCTION

_ANSWER_INSTRUCTION = {
    "default":        MATH_INSTRUCTION,
    "gpqa_diamond":   "Please reason step by step, then state your final answer as a single letter (A, B, C, or D) on the last line.",
    "livecodebench":  "Please reason step by step, then write your final answer as clean Python code.",
}


def _answer_instruction(benchmark: str) -> str:
    return _ANSWER_INSTRUCTION.get(benchmark, _ANSWER_INSTRUCTION["default"])


def _method_user_content(problem: str, i: int, benchmark: str) -> str:
    """i=1: no prefix (standard mode). i>1: [Attempt #i — use a new method] prefix."""
    instruction = _answer_instruction(benchmark)
    if i <= 1:
        return f"{problem}\n\n{instruction}"
    prefix = ATTEMPT_PREFIX.format(i=i)
    return f"{prefix}\n\n{problem}\n\n{instruction}"


def _no_system_chat(user: str) -> str:
    """Chat format without system message — matches training (no system prompt)."""
    return (
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _run_vllm_worker(
    model: str,
    lora_path: Optional[str],
    prompts: dict[str, str],  # id -> prompt text
    n: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, list[str]]:
    """Run vllm_worker as subprocess. Returns id -> list[n rollouts]."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prompts_file = os.path.join(tmpdir, "prompts.jsonl")
        out_file = os.path.join(tmpdir, "results.jsonl")

        with open(prompts_file, "w") as f:
            for pid, prompt in prompts.items():
                f.write(json.dumps({"id": pid, "prompt": prompt}) + "\n")

        cmd = [
            sys.executable, "-m", "evaluation.vllm_worker",
            "--model", model,
            "--prompts-file", prompts_file,
            "--out-file", out_file,
            "--n", str(n),
            "--temperature", str(temperature),
            "--max-tokens", str(max_tokens),
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
        return rollouts


def _pass_at_k(n_correct: int, n_total: int, k: int) -> float:
    """Unbiased pass@k estimator."""
    if n_total < k:
        return float(n_correct > 0)
    if n_total == n_correct:
        return 1.0
    return 1.0 - float(
        np.prod([(n_total - n_correct - i) / (n_total - i) for i in range(k)
                 if (n_total - n_correct - i) > 0])
    )


def _score_single(response: str, answer: str, benchmark: str) -> bool:
    from generation.rollout import extract_answer
    if benchmark == "aime24":
        try:
            return int(extract_answer(response).strip()) == int(answer.strip())
        except Exception:
            return extract_answer(response).strip() == answer.strip()
    return extract_answer(response) == answer


def eval_method_passk(
    model: str,
    lora_path: Optional[str],
    problems: list[dict],
    benchmark: str,
    max_k: int = 32,
    max_tokens: int = 4096,
    temperature: float = 0.8,
) -> dict:
    """
    Method-prompted pass@k: for each problem generate max_k samples,
    one per method_id (approach #1..#max_k).
    """
    print(f"\n[method_eval] {benchmark}: building {len(problems)} × {max_k} method prompts ...", flush=True)

    # Build prompts: one per (problem, method_id)
    # Prefix format MUST match rl/diversity_grpo.py training exactly.
    prompts: dict[str, str] = {}
    for p in problems:
        for mid in range(1, max_k + 1):
            key = f"{p['id']}__m{mid}"
            user_content = _method_user_content(p["problem"], i=mid, benchmark=benchmark)
            prompts[key] = _no_system_chat(user_content)

    print(f"[method_eval] Running vLLM for {len(prompts)} prompts (n=1 each) ...", flush=True)
    raw = _run_vllm_worker(model, lora_path, prompts, n=1, temperature=temperature, max_tokens=max_tokens)

    # Aggregate per problem: method_id -> response
    by_problem: dict[str, list[str]] = {p["id"]: [] for p in problems}
    for p in problems:
        for mid in range(1, max_k + 1):
            key = f"{p['id']}__m{mid}"
            resp = raw.get(key, [""])[0]
            by_problem[p["id"]].append(resp)

    # Score
    answer_map = {p["id"]: p["answer"] for p in problems}
    per_problem_correct: dict[str, list[bool]] = {}
    for p in problems:
        responses = by_problem[p["id"]]
        per_problem_correct[p["id"]] = [
            _score_single(r, answer_map[p["id"]], benchmark) for r in responses
        ]

    # Compute pass@k
    results: dict[str, float] = {}
    for k in [1, 2, 4, 8, 16, 32]:
        if k > max_k:
            continue
        vals = [
            _pass_at_k(sum(c[:k]), k, k)
            for c in per_problem_correct.values()
        ]
        results[f"pass_at_{k}"] = float(np.mean(vals))
        print(f"  [method] pass@{k:2d} = {results[f'pass_at_{k}']:.4f}", flush=True)

    return {
        "mode": "method",
        "benchmark": benchmark,
        "n_problems": len(problems),
        "max_k": max_k,
        "temperature": temperature,
        "metrics": results,
        "per_problem_n_correct": {pid: int(sum(c)) for pid, c in per_problem_correct.items()},
    }


def eval_standard_passk(
    model: str,
    lora_path: Optional[str],
    problems: list[dict],
    benchmark: str,
    n: int = 32,
    max_tokens: int = 4096,
    temperature: float = 0.8,
) -> dict:
    """
    Standard pass@k: single prompt per problem, n independent completions.
    Comparison baseline showing temperature-only diversity.
    """
    print(f"\n[standard_eval] {benchmark}: n={n} per problem ...", flush=True)

    prompts: dict[str, str] = {}
    for p in problems:
        instruction = _answer_instruction(benchmark)
        prompts[p["id"]] = _no_system_chat(f"{p['problem']}\n\n{instruction}")

    raw = _run_vllm_worker(model, lora_path, prompts, n=n, temperature=temperature, max_tokens=max_tokens)

    answer_map = {p["id"]: p["answer"] for p in problems}
    per_problem_correct: dict[str, list[bool]] = {}
    for p in problems:
        responses = raw.get(p["id"], [])
        per_problem_correct[p["id"]] = [
            _score_single(r, answer_map[p["id"]], benchmark) for r in responses
        ]

    results: dict[str, float] = {}
    for k in [1, 2, 4, 8, 16, 32]:
        if not all(len(c) >= k for c in per_problem_correct.values()):
            continue
        vals = [_pass_at_k(sum(c), len(c), k) for c in per_problem_correct.values()]
        results[f"pass_at_{k}"] = float(np.mean(vals))
        print(f"  [standard] pass@{k:2d} = {results[f'pass_at_{k}']:.4f}", flush=True)

    return {
        "mode": "standard",
        "benchmark": benchmark,
        "n_problems": len(problems),
        "n_samples": n,
        "temperature": temperature,
        "metrics": results,
        "per_problem_n_correct": {pid: int(sum(c)) for pid, c in per_problem_correct.items()},
    }


def run_full_eval(
    experiment_id: str,
    model: str,
    lora_path: Optional[str],
    benchmarks: list[str],
    max_k: int = 32,
    max_tokens: int = 4096,
    out_path: Optional[str] = None,
) -> dict:
    from evaluation.vllm_generate import load_benchmark

    out_path = out_path or f"results/passk/{experiment_id}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    if Path(out_path).exists():
        print(f"[passk_eval] Cached result found: {out_path} — skipping", flush=True)
        return json.loads(Path(out_path).read_text())

    print(f"\n{'='*60}", flush=True)
    print(f"[passk_eval] Experiment: {experiment_id}", flush=True)
    print(f"  model:     {model}", flush=True)
    print(f"  lora:      {lora_path or 'none (base)'}", flush=True)
    print(f"  benchmarks: {benchmarks}", flush=True)
    print(f"  max_k:     {max_k}", flush=True)
    print(f"{'='*60}", flush=True)

    result = {
        "experiment_id": experiment_id,
        "model": model,
        "lora_path": lora_path,
        "max_k": max_k,
        "benchmarks": {},
    }

    for bmark in benchmarks:
        print(f"\n[passk_eval] Benchmark: {bmark}", flush=True)
        problems = load_benchmark(bmark)
        print(f"  {len(problems)} problems loaded", flush=True)

        method_res = eval_method_passk(
            model, lora_path, problems, bmark,
            max_k=max_k, max_tokens=max_tokens, temperature=0.8,
        )
        standard_res = eval_standard_passk(
            model, lora_path, problems, bmark,
            n=max_k, max_tokens=max_tokens, temperature=0.8,
        )

        result["benchmarks"][bmark] = {
            "method": method_res,
            "standard": standard_res,
        }

        # Print comparison table
        print(f"\n  --- {bmark} pass@k comparison ---", flush=True)
        print(f"  {'k':>4}  {'method':>8}  {'standard':>8}  {'delta':>8}", flush=True)
        for k in [1, 2, 4, 8, 16, 32]:
            mk = method_res["metrics"].get(f"pass_at_{k}")
            sk = standard_res["metrics"].get(f"pass_at_{k}")
            if mk is not None and sk is not None:
                delta = mk - sk
                print(f"  {k:>4}  {mk:>8.4f}  {sk:>8.4f}  {delta:>+8.4f}", flush=True)

    Path(out_path).write_text(json.dumps(result, indent=2))
    print(f"\n[passk_eval] Results saved → {out_path}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--benchmarks", nargs="+", default=["math500", "aime24"])
    parser.add_argument("--max-k", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_full_eval(
        experiment_id=args.experiment_id,
        model=args.model,
        lora_path=args.lora_path,
        benchmarks=args.benchmarks,
        max_k=args.max_k,
        max_tokens=args.max_tokens,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
