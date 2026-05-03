"""
G1: Zero-shot diagnostic — GO/NO-GO gate for COLD-RL (i,k) prefix.

Tests whether "[PARALLEL SAMPLE i OF k]" induces diversity in the raw
base model before any fine-tuning. Run this BEFORE G3 training starts.

Pass: Arm A semantic distance > Arm B by >0.04 cosine units AND pass@8 >= Arm B.
Fail: revise prefix wording and re-run (2h max).

Usage:
    python scripts/zero_shot_diagnostic.py \
        --model Qwen/Qwen2.5-7B \
        --n_problems 100 --k 8
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams


PARALLEL_PREFIX = "[PARALLEL SAMPLE {i} OF {k}]"
SYSTEM_PROMPT = "You are a helpful assistant. Think step by step."


def build_prompts_arm_a(tokenizer, problem: str, k: int) -> list[str]:
    """k prompts with distinct role prefixes "[PARALLEL SAMPLE i OF k]"."""
    prompts = []
    for i in range(1, k + 1):
        prefix = PARALLEL_PREFIX.format(i=i, k=k)
        user_msg = f"{prefix}\n\n{problem}\n\nSolve step by step and conclude with \\boxed{{answer}}."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return prompts


def build_prompts_arm_bc(tokenizer, problem: str, k: int) -> list[str]:
    """k identical prompts, no role prefix."""
    user_msg = f"{problem}\n\nSolve step by step and conclude with \\boxed{{answer}}."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return [prompt] * k


def mean_pairwise_distance(completions: list[str], encoder: SentenceTransformer) -> float:
    if len(completions) < 2:
        return 0.0
    embs = encoder.encode(completions, normalize_embeddings=True, show_progress_bar=False)
    embs = np.array(embs)
    n = len(embs)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            cos_sim = float(np.dot(embs[i], embs[j]))
            dists.append(1.0 - cos_sim)
    return float(np.mean(dists))


def pass_at_k(completions: list[str], answer: str) -> bool:
    """Returns True if any completion contains the correct boxed answer."""
    import re
    ans_clean = str(answer).strip()
    for comp in completions:
        boxed = re.findall(r"\\boxed\{([^}]*)\}", comp)
        if any(b.strip() == ans_clean for b in boxed):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--n_problems", type=int, default=100)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/g1_diagnostic.json")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"[G1] Loading MATH-500 problems...")
    ds = load_dataset("lighteval/MATH", split="test", trust_remote_code=True)
    problems = [{"problem": r["problem"], "answer": r["solution"]} for r in ds]
    problems = random.sample(problems, min(args.n_problems, len(problems)))
    print(f"[G1] Sampled {len(problems)} problems")

    print(f"[G1] Loading model {args.model}...")
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, trust_remote_code=True)
    tokenizer = llm.get_tokenizer()

    params_07 = SamplingParams(temperature=0.7, max_tokens=2048, stop=["<|im_end|>"])
    params_11 = SamplingParams(temperature=1.1, max_tokens=2048, stop=["<|im_end|>"])

    print(f"[G1] Building prompts for {len(problems)} problems x 3 arms x k={args.k}...")
    all_a, all_b, all_c = [], [], []
    meta = []

    for prob in problems:
        all_a.extend(build_prompts_arm_a(tokenizer, prob["problem"], args.k))
        all_b.extend(build_prompts_arm_bc(tokenizer, prob["problem"], args.k))
        all_c.extend(build_prompts_arm_bc(tokenizer, prob["problem"], args.k))
        meta.append(prob["answer"])

    print("[G1] Generating Arm A (parallel prefixes, tau=0.7)...")
    outs_a = llm.generate(all_a, params_07)
    print("[G1] Generating Arm B (no prefix, tau=0.7)...")
    outs_b = llm.generate(all_b, params_07)
    print("[G1] Generating Arm C (no prefix, tau=1.1)...")
    outs_c = llm.generate(all_c, params_11)

    completions_a = [o.outputs[0].text for o in outs_a]
    completions_b = [o.outputs[0].text for o in outs_b]
    completions_c = [o.outputs[0].text for o in outs_c]

    print("[G1] Loading E5 encoder...")
    encoder = SentenceTransformer("intfloat/e5-large-instruct", device="cuda")

    div_a, div_b, div_c = [], [], []
    pass8_a, pass8_b, pass8_c = [], [], []

    for i, ans in enumerate(meta):
        grp_a = completions_a[i * args.k: (i + 1) * args.k]
        grp_b = completions_b[i * args.k: (i + 1) * args.k]
        grp_c = completions_c[i * args.k: (i + 1) * args.k]
        div_a.append(mean_pairwise_distance(grp_a, encoder))
        div_b.append(mean_pairwise_distance(grp_b, encoder))
        div_c.append(mean_pairwise_distance(grp_c, encoder))
        pass8_a.append(pass_at_k(grp_a, ans))
        pass8_b.append(pass_at_k(grp_b, ans))
        pass8_c.append(pass_at_k(grp_c, ans))

    results = {
        "arm_a_prefix_tau07": {"mean_div": float(np.mean(div_a)), "pass_at_k": float(np.mean(pass8_a))},
        "arm_b_nopfx_tau07":  {"mean_div": float(np.mean(div_b)), "pass_at_k": float(np.mean(pass8_b))},
        "arm_c_nopfx_tau11":  {"mean_div": float(np.mean(div_c)), "pass_at_k": float(np.mean(pass8_c))},
        "delta_div_AminusB":  float(np.mean(div_a)) - float(np.mean(div_b)),
    }

    threshold = 0.04
    go = (results["delta_div_AminusB"] > threshold and
          results["arm_a_prefix_tau07"]["pass_at_k"] >= results["arm_b_nopfx_tau07"]["pass_at_k"])

    results["verdict"] = "GO" if go else "NO-GO"
    results["threshold_used"] = threshold

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"[G1] RESULT: {results['verdict']}")
    print(f"  Arm A (prefix, tau=0.7):  div={results['arm_a_prefix_tau07']['mean_div']:.4f}  pass@{args.k}={results['arm_a_prefix_tau07']['pass_at_k']:.3f}")
    print(f"  Arm B (none,   tau=0.7):  div={results['arm_b_nopfx_tau07']['mean_div']:.4f}  pass@{args.k}={results['arm_b_nopfx_tau07']['pass_at_k']:.3f}")
    print(f"  Arm C (none,   tau=1.1):  div={results['arm_c_nopfx_tau11']['mean_div']:.4f}  pass@{args.k}={results['arm_c_nopfx_tau11']['pass_at_k']:.3f}")
    print(f"  Delta div (A-B): {results['delta_div_AminusB']:.4f}  (need > {threshold})")
    if results["verdict"] == "NO-GO":
        print("\n  ACTION: Revise prefix to:")
        print('  "[PARALLEL SAMPLE {i} OF {k}: use a different solution strategy')
        print('   than other parallel samples would use for this problem type]"')
        print("  Then re-run this diagnostic.")
    print("=" * 60)


if __name__ == "__main__":
    main()
