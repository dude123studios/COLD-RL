"""
Test-Time Scaling via wait-token injection (budget forcing).

For each problem:
  1. Generate a response (max_tokens budget)
  2. If no answer found, append a wait token and continue generating
  3. Repeat up to `n_rounds` times

This tests whether PAS-trained models benefit more from extended compute
than SFT baselines, as they've seen diverse solution strategies.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from evaluation.vllm_generate import (
    load_benchmark, _prompt, _cache_key, EVAL_CACHE_DIR,
    score_answers, extract_choice, MC_BENCHMARKS
)
from generation.rollout import extract_answer

WAIT_TOKENS = {
    "qwen":    "\n\nWait, let me reconsider.",
    "llama":   "\n\nWait, let me reconsider.",
    "mistral": "\n\nWait, let me reconsider.",
    "gemma":   "\n\nWait, let me reconsider.",
    "default": "\n\nWait, let me reconsider.",
}

def _wait_token(model: str) -> str:
    m = model.lower()
    for k in WAIT_TOKENS:
        if k in m:
            return WAIT_TOKENS[k]
    return WAIT_TOKENS["default"]


def _has_answer(text: str, benchmark: str) -> bool:
    if benchmark in MC_BENCHMARKS:
        return bool(extract_choice(text))
    return bool(extract_answer(text))


def _vllm_generate_prompts(
    base_model: str,
    lora_path: Optional[str],
    prompts: dict[str, str],  # id -> prompt text
    n: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, list[str]]:
    """Run vllm_worker on arbitrary prompts. Returns id -> list[str] rollouts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prompts_file = os.path.join(tmpdir, "prompts.jsonl")
        out_file = os.path.join(tmpdir, "results.jsonl")

        with open(prompts_file, "w") as f:
            for pid, prompt in prompts.items():
                f.write(json.dumps({"id": pid, "prompt": prompt}) + "\n")

        cmd = [
            sys.executable, "-m", "evaluation.vllm_worker",
            "--model", base_model,
            "--prompts-file", prompts_file,
            "--out-file", out_file,
            "--n", str(n),
            "--temperature", str(temperature),
            "--max-tokens", str(max_tokens),
        ]
        if lora_path:
            cmd += ["--lora-path", lora_path]

        env = os.environ.copy()
        repo_dir = str(Path(__file__).parent.parent)
        env["PYTHONPATH"] = repo_dir + ":" + env.get("PYTHONPATH", "")

        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"vllm_worker exited with code {result.returncode}")

        rollouts = {}
        with open(out_file) as f:
            for line in f:
                row = json.loads(line)
                rollouts[row["id"]] = row["rollouts"]
        return rollouts


def tts_eval(
    base_model: str,
    lora_path: Optional[str],
    benchmark: str,
    n: int = 1,
    n_rounds: int = 3,
    max_tokens_per_round: int = 2048,
    temperature: float = 0.6,
) -> dict:
    """
    Budget-forcing TTS: inject wait tokens for problems without answers.

    Returns dict with:
      - scores per round (pass_at_1 after each wait injection)
      - final rollouts dict (problem_id -> list of final texts)
    """
    problems = load_benchmark(benchmark)
    wait_tok = _wait_token(base_model)

    # Current prompt for each sample: (problem_id, sample_idx) -> full prompt
    # We run n independent samples per problem
    prompts: dict[str, str] = {}
    for p in problems:
        for s in range(n):
            key = f"{p['id']}__s{s}"
            prompts[key] = _prompt(p["problem"], benchmark, base_model)

    # Accumulated text per sample
    texts: dict[str, str] = {k: "" for k in prompts}
    answered: dict[str, bool] = {k: False for k in prompts}

    round_scores = []

    for rnd in range(n_rounds + 1):
        pending_keys = [k for k in prompts if not answered[k]]
        if not pending_keys:
            break

        print(f"[tts] round {rnd}: generating {len(pending_keys)} samples "
              f"({'initial' if rnd == 0 else f'wait injection #{rnd}'})")

        pending_prompts = {k: prompts[k] for k in pending_keys}
        rollouts = _vllm_generate_prompts(
            base_model, lora_path,
            pending_prompts,
            n=1,
            temperature=temperature,
            max_tokens=max_tokens_per_round,
        )

        for k in pending_keys:
            gen = rollouts.get(k, [""])[0]
            texts[k] += gen
            if _has_answer(texts[k], benchmark):
                answered[k] = True
            elif rnd < n_rounds:
                # Inject wait token and continue
                prompts[k] = prompts[k] + texts[k] + wait_tok
                texts[k] += wait_tok

        # Score at this round
        rollouts_by_pid: dict[str, list[str]] = {}
        for p in problems:
            rollouts_by_pid[p["id"]] = [
                texts[f"{p['id']}__s{s}"] for s in range(n)
            ]
        scores = score_answers(rollouts_by_pid, problems, benchmark=benchmark)
        round_scores.append({"round": rnd, "wait_injections": rnd, **scores})
        print(f"[tts] round {rnd} scores: {scores}")

    # Build final rollouts (collapse back to problem-level)
    final_rollouts: dict[str, list[str]] = {}
    for p in problems:
        final_rollouts[p["id"]] = [texts[f"{p['id']}__s{s}"] for s in range(n)]

    return {
        "round_scores": round_scores,
        "final_scores": round_scores[-1] if round_scores else {},
        "rollouts": final_rollouts,
    }


def run_tts_experiment(
    experiment_id: str,
    base_model: str,
    lora_path: Optional[str],
    benchmarks: list[str],
    n: int = 4,
    n_rounds: int = 3,
    max_tokens_per_round: int = 2048,
    temperature: float = 0.6,
    results_dir: str = "results",
) -> dict:
    """Run TTS eval and save results."""
    out_path = Path(results_dir) / f"tts_{experiment_id}.json"
    if out_path.exists():
        print(f"[tts] Cached: {out_path}")
        return json.loads(out_path.read_text())

    results = {"experiment_id": experiment_id, "base_model": base_model}
    for bmark in benchmarks:
        print(f"[tts] {experiment_id} — {bmark} (n={n}, rounds={n_rounds})")
        res = tts_eval(
            base_model, lora_path, bmark,
            n=n, n_rounds=n_rounds,
            max_tokens_per_round=max_tokens_per_round,
            temperature=temperature,
        )
        results[bmark] = res["round_scores"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[tts] Saved to {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--benchmarks", nargs="+", default=["math500", "gpqa_diamond"])
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()

    run_tts_experiment(
        experiment_id=args.experiment_id,
        base_model=args.model,
        lora_path=args.lora_path,
        benchmarks=args.benchmarks,
        n=args.n,
        n_rounds=args.rounds,
        max_tokens_per_round=args.max_tokens,
        temperature=args.temperature,
    )
