"""
Rollout generation for student model evaluation.
Uses vLLM for fast batched inference.
Caches results keyed by (model, problem_id, n, temperature).
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Optional

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

CACHE_DIR = Path(__file__).parent.parent / "results" / "rollout_cache"


def _cache_key(model_name: str, problem_id: str, n: int, temperature: float) -> str:
    raw = f"{model_name}|{problem_id}|{n}|{temperature}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def extract_answer(text: str) -> str:
    """
    Priority order:
    1. \\boxed{...}
    2. "the answer is X" pattern
    3. Last number/expression on last non-empty line
    """
    # 1. Boxed
    boxed = re.findall(r"\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", text)
    if boxed:
        return _normalize(boxed[-1])

    # 2. "the answer is" pattern
    m = re.search(
        r"(?:the answer is|answer:|=)\s*\$?([^\n$]+)\$?",
        text,
        re.IGNORECASE,
    )
    if m:
        return _normalize(m.group(1).strip())

    # 3. Last number on last non-empty line
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", lines[-1])
        if nums:
            return _normalize(nums[-1])

    return ""


def _normalize(ans: str) -> str:
    ans = ans.strip().rstrip(".")
    # Strip surrounding dollar signs
    ans = re.sub(r"^\$|\$$", "", ans).strip()
    return ans


def generate_rollouts(
    model_name: str,
    problems: list[dict],
    n: int = 16,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    save_path: Optional[str] = None,
    use_cache: bool = True,
    lora_path: Optional[str] = None,
) -> dict[str, list[str]]:
    """
    Generate n rollouts per problem using vLLM.

    Args:
        model_name: HF model path or local path
        problems: list of {id, problem, answer}
        n: samples per problem
        temperature: sampling temperature
        max_tokens: max new tokens
        save_path: if set, cache JSONL path
        use_cache: load from cache if available
        lora_path: optional LoRA adapter path

    Returns:
        dict mapping problem_id -> list of n rollout strings
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = save_path or str(
        CACHE_DIR / f"{_cache_key(model_name, 'all', n, temperature)}.jsonl"
    )

    # Load existing cache
    cached: dict[str, list[str]] = {}
    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                row = json.loads(line)
                cached[row["id"]] = row["rollouts"]

    missing = [p for p in problems if p["id"] not in cached]
    if not missing:
        return cached

    # Build prompts
    system_prompt = (
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
    prompts = []
    for p in missing:
        prompts.append(
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{p['problem']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    # Load model
    llm_kwargs = dict(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.70,
        max_model_len=8192,
        enforce_eager=True,
        disable_log_stats=True,
        dtype="bfloat16",
    )
    if lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        n=n,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    lora_req = None
    if lora_path:
        lora_req = LoRARequest("adapter", 1, lora_path)

    outputs = llm.generate(prompts, sampling, lora_request=lora_req)

    # Collect and cache results
    results = dict(cached)
    with open(cache_path, "a") as f:
        for prob, output in zip(missing, outputs):
            rollouts = [o.text for o in output.outputs]
            results[prob["id"]] = rollouts
            f.write(json.dumps({"id": prob["id"], "rollouts": rollouts}) + "\n")

    del llm
    return results
