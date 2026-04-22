"""
Eval generation using vLLM with optional LoRA adapter.
Dispatches to the correct benchmark loader and answer extractor.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from generation.rollout import extract_answer

EVAL_CACHE_DIR = Path(__file__).parent.parent / "results" / "eval_cache"

SYSTEM_PROMPT_MATH = "Please reason step by step, and put your final answer within \\boxed{}."
SYSTEM_PROMPT_GPQA = "Please reason step by step, then state your final answer as a single letter (A, B, C, or D) on the last line."
SYSTEM_PROMPT_CODE = "You are an expert competitive programmer. Write clean, efficient Python code."

# Backwards-compatible alias
SYSTEM_PROMPT = SYSTEM_PROMPT_MATH

MC_BENCHMARKS = {"gpqa_diamond", "mmlu", "arc"}


def _format_chat(content: str, system: str, model: str = "") -> str:
    m = model.lower()
    if "llama-3" in m or "llama3" in m:
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{content}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif "mistral" in m or "mixtral" in m:
        return f"[INST] {system}\n\n{content} [/INST]"
    elif "gemma" in m:
        return (
            f"<start_of_turn>user\n{system}\n\n{content}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    else:  # Qwen2/2.5, default
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


def _prompt(problem: str, benchmark: str = "math500", model: str = "") -> str:
    if benchmark in MC_BENCHMARKS:
        sys = SYSTEM_PROMPT_GPQA
    elif benchmark == "livecodebench":
        sys = SYSTEM_PROMPT_CODE
    else:
        sys = SYSTEM_PROMPT_MATH
    return _format_chat(problem, sys, model)


def _cache_key(base_model: str, lora_path: Optional[str], benchmark: str, n: int, temperature: float) -> str:
    raw = f"{base_model}|{lora_path or 'none'}|{benchmark}|{n}|{temperature}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def load_benchmark(benchmark: str, extra_problems: list[dict] = None) -> list[dict]:
    """
    Load benchmark problems. Returns list of {id, problem, answer}.

    Args:
        benchmark: "math500", or a path to a custom .jsonl file
        extra_problems: optional list of extra {id, problem, answer} dicts to append
                        (used for rare-problem family evaluation)
    """
    data_dir = Path(__file__).parent.parent / "data"

    if benchmark == "math500":
        from generation.build_sft_dataset import load_math500
        problems = load_math500()
    elif benchmark in ("gpqa_diamond", "livecodebench", "aime24", "mmlu", "arc", "gsm8k", "amc23", "math_l1", "math_l2", "math_l3", "math_l4"):
        path = data_dir / "benchmarks" / f"{benchmark}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Benchmark not found: {path}\n"
                f"Run: python -m scripts.download_benchmarks"
            )
        problems = []
        with open(path) as f:
            for line in f:
                problems.append(json.loads(line))
    else:
        p = Path(benchmark)
        path = p if p.is_absolute() else data_dir / f"{benchmark}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Benchmark not found: {path}")
        problems = []
        with open(path) as f:
            for line in f:
                problems.append(json.loads(line))

    if extra_problems:
        problems = problems + extra_problems

    return problems


def extract_choice(text: str) -> str:
    """
    Extract a multiple-choice letter (A/B/C/D) from a model response.
    Priority: explicit final answer markers, then last standalone letter.
    """
    # "The answer is (B)" / "Answer: C" / "**D**"
    for pattern in [
        r"(?:answer is|answer:|final answer)[^\w]*([ABCD])\b",
        r"\b([ABCD])\)\s*$",
        r"\*\*([ABCD])\*\*",
        r"\(([ABCD])\)",
    ]:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    # Last standalone letter on last non-empty line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        m = re.search(r"^([ABCD])\.?$", lines[-1], re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return ""


def score_answers(
    rollouts: dict[str, list[str]],
    problems: list[dict],
    benchmark: str = "math500",
) -> dict:
    """
    Compute pass@k metrics for k in {1, 4, 8, 16}.

    Dispatcher:
      - livecodebench: execution-based scoring via code_scorer
      - gpqa_diamond:  letter extraction (A/B/C/D)
      - everything else: boxed math answer extraction
    """
    if benchmark == "livecodebench":
        from evaluation.code_scorer import score_livecodebench
        return score_livecodebench(rollouts, problems)

    import numpy as np

    answer_map = {p["id"]: p["answer"] for p in problems}
    extractor = extract_choice if benchmark in MC_BENCHMARKS else extract_answer

    def _answers_match(pred: str, gold: str, benchmark: str) -> bool:
        if benchmark == "aime24":
            # Normalize integers: "023" == "23"
            try:
                return int(pred.strip()) == int(gold.strip())
            except (ValueError, AttributeError):
                return pred.strip() == gold.strip()
        return pred == gold

    correct_by_problem = []
    for p in problems:
        pid = p["id"]
        gold = answer_map[pid]
        rolls = rollouts.get(pid, [])
        correct = [_answers_match(extractor(r), gold, benchmark) for r in rolls]
        correct_by_problem.append(correct)

    def pass_at_k(correct_list: list[bool], k: int) -> float:
        n = len(correct_list)
        c = sum(correct_list)
        if n < k:
            return float(c > 0)
        if n == c:
            return 1.0
        return 1.0 - float(np.prod([(n - c - i) / (n - i) for i in range(k) if n - c - i > 0]))

    results = {}
    for k in [1, 4, 8, 16, 32, 64]:
        if all(len(c) >= k for c in correct_by_problem):
            results[f"pass_at_{k}"] = float(
                np.mean([pass_at_k(c, k) for c in correct_by_problem])
            )

    return results


def eval_generate(
    base_model: str,
    lora_path: Optional[str],
    benchmark: str,
    n: int = 16,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    use_cache: bool = True,
    extra_problems: Optional[list[dict]] = None,
) -> dict[str, list[str]]:
    """
    Generate n rollouts per problem on a benchmark using vLLM.

    Args:
        base_model: HF model name or local path
        lora_path: path to LoRA adapter directory (or None for base model)
        benchmark: benchmark name ("math500", etc.)
        n: rollouts per problem
        temperature: sampling temperature
        max_tokens: max new tokens per rollout
        use_cache: load from cache if available

    Returns:
        dict mapping problem_id -> list of n rollout strings
    """
    EVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = EVAL_CACHE_DIR / f"{_cache_key(base_model, lora_path, benchmark, n, temperature)}.jsonl"

    # Load cache
    cached: dict[str, list[str]] = {}
    if use_cache and cache_file.exists():
        with open(cache_file) as f:
            for line in f:
                row = json.loads(line)
                cached[row["id"]] = row["rollouts"]

    problems = load_benchmark(benchmark, extra_problems=extra_problems)
    missing = [p for p in problems if p["id"] not in cached]

    if not missing:
        print(f"[eval] {benchmark}: all {len(problems)} problems cached")
        return {p["id"]: cached[p["id"]] for p in problems}

    print(f"[eval] {benchmark}: generating {len(missing)} missing problems (n={n})", flush=True)

    # Run vLLM in a subprocess so that vLLM's atexit/os._exit() cannot kill
    # the main orchestrator process.
    with tempfile.TemporaryDirectory() as tmpdir:
        prompts_file = os.path.join(tmpdir, "prompts.jsonl")
        out_file = os.path.join(tmpdir, "results.jsonl")

        with open(prompts_file, "w") as f:
            for p in missing:
                f.write(json.dumps({"id": p["id"], "prompt": _prompt(p["problem"], benchmark, base_model)}) + "\n")

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

        with open(cache_file, "a") as cf:
            with open(out_file) as rf:
                for line in rf:
                    row = json.loads(line)
                    cached[row["id"]] = row["rollouts"]
                    cf.write(line)

    print(f"[eval] {benchmark}: done, results cached", flush=True)
    return {p["id"]: cached[p["id"]] for p in problems}
