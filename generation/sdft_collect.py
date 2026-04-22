"""
SDFT (Self-Distillation Fine-Tuning) data collection.

For each problem:
  1. Student rollout: model generates a trace with no solution context
  2. Teacher logits: same model, conditioned on problem + solution, run as a
     single causal forward pass over the student's tokens to get P(token | context)
     at every position — equivalent to the "prefill 1-token-at-a-time" procedure
     described in the paper but executed in one efficient batched forward pass.
  3. Top-K compression: only keep K=50 logit values/ids per position
  4. Save per-problem numpy arrays + JSONL index

Cache layout:
  results/sdft_cache/{experiment_id}/
    index.jsonl                        — one line per completed problem
    {problem_id}_tokens.npy            — int32 [T]
    {problem_id}_vals.npy              — float16 [T, K]
    {problem_id}_ids.npy               — int32 [T, K]
"""

import gc
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

SDFT_CACHE_DIR = Path(__file__).parent.parent / "results" / "sdft_cache"

SYSTEM_MSG = (
    "You are an expert math tutor. When solving problems, think out loud: "
    "explore the problem structure, consider multiple approaches, check your "
    "work as you go, and explain your reasoning at every step."
)

STUDENT_USER_SUFFIX = (
    "\n\nThink through this carefully. Explore the problem, consider different "
    "approaches, and show your full reasoning before giving the final answer."
)


def _build_student_prompt(tokenizer, problem: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": problem + STUDENT_USER_SUFFIX},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _build_teacher_prompt(tokenizer, problem: str, solution: str) -> str:
    content = (
        f"{problem}\n\n"
        f"Here is a complete solution showing the key approach:\n{solution}\n\n"
        f"Now solve this problem yourself, thinking step by step. You can use the "
        f"same approach or a different one — show all your reasoning and work."
    )
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def collect_sdft_data(
    model_name: str,
    problems: list[dict],
    max_new_tokens: int = 8192,
    top_k: int = 50,
    temperature: float = 0.6,
    experiment_id: str = "default",
) -> str:
    """
    Collect student rollouts + teacher logits for all problems.

    Args:
        model_name: HuggingFace model name
        problems: list of {id, problem, solution, answer}
        max_new_tokens: max generation length for student rollout
        top_k: number of top logits to save per token position
        temperature: sampling temperature for student rollout
        experiment_id: used for cache directory naming

    Returns:
        Path to cache directory (str)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = SDFT_CACHE_DIR / experiment_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "index.jsonl"

    # Load completed ids from index
    done_ids: set[str] = set()
    if index_path.exists():
        with open(index_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

    todo = [p for p in problems if p["id"] not in done_ids]
    if not todo:
        print(f"[SDFT] All {len(problems)} problems already cached at {cache_dir}")
        return str(cache_dir)

    print(f"[SDFT] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    device = next(model.parameters()).device

    index_file = open(index_path, "a")
    try:
        for i, prob in enumerate(todo):
            pid = prob["id"]

            # ── Student rollout ────────────────────────────────────────────
            student_text = _build_student_prompt(tokenizer, prob["problem"])
            student_enc = tokenizer(
                student_text, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)

            with torch.no_grad():
                gen_out = model.generate(
                    student_enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id,
                )

            student_tokens = gen_out[0, student_enc.shape[1]:]  # [T]
            T = student_tokens.shape[0]
            if T == 0:
                continue

            # ── Teacher forward pass ───────────────────────────────────────
            # Single causal forward pass over [teacher_prefix + student_tokens].
            # teacher_logits[P-1 : P+T-1] predicts student_tokens[0 : T].
            teacher_text = _build_teacher_prompt(
                tokenizer, prob["problem"], prob["solution"]
            )
            teacher_enc = tokenizer(
                teacher_text, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            P = teacher_enc.shape[1]

            teacher_input = torch.cat(
                [teacher_enc, student_tokens.unsqueeze(0)], dim=1
            )

            with torch.no_grad():
                teacher_out = model(teacher_input)
                # logits[P-1] predicts student_tokens[0], etc.
                logits = teacher_out.logits[0, P - 1 : P + T - 1, :]  # [T, vocab]

            # Top-K compression (float16 values, int32 ids)
            topk_vals, topk_ids = torch.topk(logits.float(), k=top_k, dim=-1)
            topk_vals_np = topk_vals.half().cpu().numpy().astype(np.float16)  # [T, K]
            topk_ids_np = topk_ids.cpu().numpy().astype(np.int32)             # [T, K]
            tokens_np = student_tokens.cpu().numpy().astype(np.int32)        # [T]

            np.save(cache_dir / f"{pid}_tokens.npy", tokens_np)
            np.save(cache_dir / f"{pid}_vals.npy", topk_vals_np)
            np.save(cache_dir / f"{pid}_ids.npy", topk_ids_np)

            record = {
                "id": pid,
                "T": int(T),
                "answer": prob["answer"],
                "problem": prob["problem"],
            }
            index_file.write(json.dumps(record) + "\n")
            index_file.flush()

            if (i + 1) % 50 == 0 or i == 0:
                print(
                    f"[SDFT] {i+1}/{len(todo)} collected  "
                    f"(T={T}, skip_rate={len(done_ids)}/{len(problems)})",
                    flush=True,
                )

            # Free intermediate tensors aggressively
            del gen_out, student_tokens, teacher_input, teacher_out, logits
            del topk_vals, topk_ids
            torch.cuda.empty_cache()

    finally:
        index_file.close()

    print(f"[SDFT] Collection done — {len(todo)} new problems → {cache_dir}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[SDFT] GPU freed: {free/1e9:.1f}/{total/1e9:.1f} GiB free")

    return str(cache_dir)
