"""
Build SFT training datasets for every baseline and ablation condition.

All dataset variants are built OFFLINE before any training starts.
Each row includes an experiment_id and source tag for ablation analysis.

Training data source: 12.5k MATH dataset (Hendrycks et al.)
Eval: MATH-500
"""

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import load_dataset

from generation.pas_augment import PASAugmenter
from generation.rollout import extract_answer

DATA_DIR = Path(__file__).parent.parent / "data"
DATASET_DIR = Path(__file__).parent.parent / "results" / "datasets"

MATH_SPLIT_TRAIN = "train"  # 12.5k problems
MATH_HF_PATH = "qwedsacf/competition_math"

CHAT_TEMPLATE = (
    "<|im_start|>system\nPlease reason step by step, and put your final answer "
    "within \\boxed{{}}.<|im_end|>\n"
    "<|im_start|>user\n{problem}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

SYSTEM_PROMPT_MATH = "Please reason step by step, and put your final answer within \\boxed{}."


def _chat_prompt(problem: str, model_name: str = "") -> str:
    m = model_name.lower()
    if "llama-3" in m or "llama3" in m:
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT_MATH}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{problem}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif "mistral" in m or "mixtral" in m:
        return f"[INST] {SYSTEM_PROMPT_MATH}\n\n{problem} [/INST]"
    elif "gemma" in m:
        return (
            f"<start_of_turn>user\n{SYSTEM_PROMPT_MATH}\n\n{problem}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    else:  # Qwen2/2.5, default
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT_MATH}<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


def _extract_gsm8k_answer(solution: str) -> str:
    """Extract numeric answer from GSM8K '#### <number>' format."""
    import re
    match = re.search(r'####\s*([\d,]+(?:\.\d+)?)', solution)
    if match:
        return match.group(1).replace(',', '')
    return extract_answer(solution)


def _load_gsm8k_problems() -> list[dict]:
    """Load GSM8K training problems (7473 examples)."""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    problems = []
    for i, row in enumerate(ds):
        problems.append(
            {
                "id": f"gsm8k_{i}",
                "problem": row["question"],
                "solution": row["answer"],
                "answer": _extract_gsm8k_answer(row["answer"]),
                "level": "intermediate",
                "type": "word_problem",
            }
        )
    return problems


def _load_all_problems() -> list[dict]:
    """Load all 12.5k problems from qwedsacf/competition_math train split."""
    ds = load_dataset(MATH_HF_PATH, split=MATH_SPLIT_TRAIN)
    problems = []
    for i, row in enumerate(ds):
        problems.append(
            {
                "id": f"math_{i}",
                "problem": row["problem"],
                "solution": row["solution"],
                "answer": extract_answer(row["solution"]),
                "level": row.get("level", ""),
                "type": row.get("type", ""),
            }
        )
    return problems


def _get_eval_ids(all_problems: list[dict], n_eval: int = 500, seed: int = 0) -> set[str]:
    """Return fixed set of problem IDs reserved for eval (never used in training)."""
    rng = random.Random(seed)
    sampled = rng.sample(all_problems, n_eval)
    return {p["id"] for p in sampled}


def load_math_train(n_problems: Optional[int] = None, seed: int = 42, dataset: str = "math") -> list[dict]:
    """
    Load training problems.

    Args:
        n_problems: Optional subsample size
        seed: Random seed for subsampling
        dataset: "math" (12.5k MATH dataset) or "gsm8k" (7473 GSM8K problems)

    Returns:
        Training problems, optionally subsampled.
    """
    if dataset == "gsm8k":
        all_problems = _load_gsm8k_problems()
        # For GSM8K, use all problems for training (no held-out eval set like MATH)
        train = all_problems
    else:
        # Default: MATH dataset (12.5k minus 500 held-out eval problems)
        all_problems = _load_all_problems()
        eval_ids = _get_eval_ids(all_problems)
        train = [p for p in all_problems if p["id"] not in eval_ids]

    if n_problems is not None and n_problems < len(train):
        rng = random.Random(seed)
        train = rng.sample(train, n_problems)

    return train


def load_math500() -> list[dict]:
    """
    Load the 500 held-out eval problems.
    Same fixed seed=0 split as load_math_train excludes, so train/eval never overlap.
    Cached to data/math500.jsonl after first load.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / "math500.jsonl"

    if cache_path.exists():
        with open(cache_path) as f:
            return [json.loads(l) for l in f]

    all_problems = _load_all_problems()
    eval_ids = _get_eval_ids(all_problems)
    eval_problems = [p for p in all_problems if p["id"] in eval_ids]

    with open(cache_path, "w") as f:
        for p in eval_problems:
            f.write(json.dumps(p) + "\n")

    return eval_problems


def _compute_embedding_diversity(solutions: list[str]) -> np.ndarray:
    """
    Compute pairwise diversity scores for selecting diverse solutions.
    Used by sft_3diverse strategy.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(solutions, normalize_embeddings=True)
    return embeddings


def _select_diverse_solutions(
    solutions: list[str], n: int = 3, seed: int = 42
) -> list[str]:
    """
    Select n maximally diverse solutions using greedy farthest-point sampling.
    """
    if len(solutions) <= n:
        return solutions

    embeddings = _compute_embedding_diversity(solutions)
    rng = random.Random(seed)
    first = rng.randint(0, len(solutions) - 1)
    selected = [first]

    while len(selected) < n:
        selected_embs = embeddings[selected]
        # For each candidate, compute min distance to any selected
        min_dists = np.min(
            1.0 - np.dot(embeddings, selected_embs.T), axis=1
        )
        min_dists[selected] = -1  # exclude already selected
        selected.append(int(np.argmax(min_dists)))

    return [solutions[i] for i in selected]


def build_dataset(
    strategy: str,
    n_problems: Optional[int] = None,
    pas_k: int = 3,
    pas_context_mode: str = "full",
    rdiv_threshold: float = 0.25,
    seed: int = 42,
    teacher_model: Optional[str] = None,
    model_name: str = "",
    dataset: str = "math",
    **kwargs,
) -> list[dict]:
    """
    Build training dataset for a given strategy.

    Strategies:
      sft_1s    — 1 solution per problem (original)
      sft_3s    — 3 solutions per problem (all original, top-3 by quality)
      sft_3diverse — 3 maximally diverse solutions (embedding-based selection)
      pas       — original + PAS-generated alternatives (k per problem)

    Args:
        dataset: "math" (MATH dataset) or "gsm8k" (GSM8K dataset)

    Returns list of HF-format training rows.
    """
    problems = load_math_train(n_problems, seed=seed, dataset=dataset)
    experiment_id = f"{strategy}_n{n_problems or 'all'}_k{pas_k}_{pas_context_mode}"

    rows = []

    if strategy == "grpo":
        # GRPO doesn't use pre-generated solutions — needs problems + answers only.
        # _make_row stores solution as "output"; for GRPO we put empty string.
        for p in problems:
            rows.append(_make_row(p["problem"], "", p["id"], "grpo", experiment_id, model_name))
            rows[-1]["answer"] = p["answer"]  # grpo_train.py reads this

    elif strategy == "sft_1s":
        for p in problems:
            rows.append(_make_row(p["problem"], p["solution"], p["id"], "original", experiment_id, model_name))

    elif strategy in ("sft_3s", "sft_3diverse", "pas", "pas_minimal_edit", "pas_v3"):
        use_local = teacher_model is None or teacher_model.startswith("Qwen/") or teacher_model.startswith("meta-llama/")
        model_for_gen = teacher_model or "Qwen/Qwen2.5-3B-Instruct"
        augmenter = PASAugmenter(teacher_model=model_for_gen, use_local=use_local)

        if strategy in ("pas_minimal_edit", "pas_v3"):
            # Phase 1: rejection-sample from STUDENT model (on-policy, in-distribution)
            # Phase 2: PAS augmentation — rewrite student solutions via TEACHER API
            #   The teacher (deepseek/deepseek-chat via OpenRouter) is ONLY used in phase 2.
            #   model_name is the student model; model_for_gen is the API teacher.
            effective_context = pas_context_mode  # use whatever was passed, not hardcoded
            teacher_rows = _build_teacher_gen_dataset(
                problems=problems,
                model_name=model_name,  # STUDENT model for rejection sampling
                experiment_id=f"teachergen_{experiment_id}",
                n_samples=8,
                temperature=0.6,
                chat_model_name=model_name,
            )
            # Use teacher-gen solutions as the base for PAS augmentation
            teacher_problems = [
                {**p, "solution": next(
                    (r["output"] for r in teacher_rows if r["problem_id"] == p["id"]), p["solution"]
                )}
                for p in problems
                if any(r["problem_id"] == p["id"] for r in teacher_rows)
            ]
            augmented = augmenter.augment_dataset(
                teacher_problems, k=pas_k, context_mode=effective_context,
                experiment_id=f"pasme_{experiment_id}", include_original=True,
            )
            for p in augmented:
                rows.append(_make_row(p["problem"], p["solution"], p["id"], p["source"], experiment_id, model_name))

        elif strategy == "sft_3s":
            augmented = augmenter.augment_dataset(
                problems, k=2, context_mode="full",
                experiment_id=f"sft3s_{experiment_id}", include_original=True,
            )
            for p in augmented:
                rows.append(_make_row(p["problem"], p["solution"], p["id"], p["source"], experiment_id, model_name))

        elif strategy == "sft_3diverse":
            augmented = augmenter.augment_dataset(
                problems, k=5, context_mode="full",
                experiment_id=f"sft3div_{experiment_id}", include_original=True,
            )
            by_id: dict[str, list[dict]] = {}
            for p in augmented:
                by_id.setdefault(p["id"], []).append(p)
            for prob_id, candidates in by_id.items():
                solutions = [c["solution"] for c in candidates]
                diverse = _select_diverse_solutions(solutions, n=3, seed=seed)
                for sol in diverse:
                    rows.append(_make_row(candidates[0]["problem"], sol, prob_id, "diverse", experiment_id, model_name))

        else:  # pas
            augmented = augmenter.augment_dataset(
                problems, k=pas_k, context_mode=pas_context_mode,
                experiment_id=f"pas_{experiment_id}", include_original=True,
            )
            for p in augmented:
                rows.append(_make_row(p["problem"], p["solution"], p["id"], p["source"], experiment_id, model_name))

    elif strategy == "teacher_gen":
        gen_model = teacher_model
        rows = _build_teacher_gen_dataset(
            problems=problems,
            model_name=gen_model,
            experiment_id=experiment_id,
            n_samples=8,
            temperature=0.6,
            chat_model_name=model_name,
        )

    elif strategy == "edf":
        from generation.edf_augment import build_edf_dataset, edf_rows_to_sft_format
        plan_model = teacher_model or "deepseek/deepseek-chat"
        trace_model = kwargs.get("trace_model", "deepseek/deepseek-r1-distill-qwen-32b")
        edf_rows = build_edf_dataset(
            problems, k=pas_k, plan_model=plan_model, trace_model=trace_model,
            experiment_id=f"edf_{experiment_id}",
            judge=kwargs.get("judge", False),
            judge_model=kwargs.get("judge_model", "deepseek/deepseek-chat"),
        )
        rows = edf_rows_to_sft_format(edf_rows, model_name=model_name)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return rows


def _build_teacher_gen_dataset(
    problems: list[dict],
    model_name: str,
    experiment_id: str,
    n_samples: int = 8,
    temperature: float = 0.6,
    chat_model_name: str = "",
) -> list[dict]:
    """
    Generate solutions using the teacher model via rejection sampling.
    For each problem: generate n_samples solutions, keep the first correct one.
    Solutions are in the model's own distribution — no textbook LaTeX gap.
    """
    from evaluation.vllm_generate import eval_generate, score_answers

    cache_path = DATASET_DIR / f"{experiment_id}_teacher_gen.jsonl"
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Load from cache if available
    cached: dict[str, str] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                row = json.loads(line)
                cached[row["problem_id"]] = row["solution"]
        print(f"[teacher_gen] Loaded {len(cached)} cached solutions")

    missing = [p for p in problems if p["id"] not in cached]

    if missing:
        print(f"[teacher_gen] Generating solutions for {len(missing)} problems with {model_name}")
        # Write problems to a temp benchmark file so eval_generate can load them
        tmp_benchmark = DATASET_DIR / f"{experiment_id}_tmp_benchmark.jsonl"
        with open(tmp_benchmark, "w") as f:
            for p in missing:
                f.write(json.dumps({"id": p["id"], "problem": p["problem"], "answer": p["answer"]}) + "\n")
        rollouts = eval_generate(
            base_model=model_name,
            lora_path=None,
            benchmark=str(tmp_benchmark),
            n=n_samples,
            temperature=temperature,
        )
        tmp_benchmark.unlink(missing_ok=True)
        # Filter: keep first correct rollout per problem
        with open(cache_path, "a") as f:
            for p in missing:
                pid = p["id"]
                candidates = rollouts.get(pid, [])
                correct = [c for c in candidates if extract_answer(c) == p["answer"]]
                if not correct:
                    continue  # skip problems where no sample was correct
                solution = correct[0]
                cached[pid] = solution
                f.write(json.dumps({"problem_id": pid, "solution": solution}) + "\n")

        n_found = sum(1 for p in problems if p["id"] in cached)
        print(f"[teacher_gen] Yield: {n_found}/{len(problems)} ({100*n_found/len(problems):.1f}%) problems have at least 1 correct solution")
        if n_found == 0:
            raise RuntimeError(
                f"[teacher_gen] Zero correct solutions found for {model_name} on {len(problems)} problems. "
                f"Model may be too weak for this task or answer extraction is broken."
            )

    rows = []
    for p in problems:
        if p["id"] not in cached:
            continue
        rows.append(_make_row(p["problem"], cached[p["id"]], p["id"], "teacher_gen", experiment_id, chat_model_name))
    return rows


def _make_row(problem: str, solution: str, problem_id: str, source: str, experiment_id: str, model_name: str = "") -> dict:
    return {
        "input": _chat_prompt(problem, model_name) if model_name else CHAT_TEMPLATE.format(problem=problem),
        "output": solution,
        "problem_id": problem_id,
        "source": source,
        "experiment_id": experiment_id,
    }


def save_dataset(rows: list[dict], path: str) -> None:
    if not rows:
        raise RuntimeError(
            f"[dataset] ABORT: attempted to save 0 rows to {path}. "
            f"Data generation produced no training examples — check API credits and logs."
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[dataset] Saved {len(rows)} rows to {path}")


def build_dataset_for_config(config: dict) -> str:
    """Build dataset for an experiment config and return the path."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    exp_id = config["id"]
    path = str(DATASET_DIR / f"{exp_id}.jsonl")

    if Path(path).exists():
        print(f"[dataset] Using cached dataset: {path}")
        return path

    training = config.get("training", {})
    strategy = training.get("strategy", "none")
    if strategy == "none":
        return ""

    rows = build_dataset(
        strategy=strategy,
        n_problems=training.get("dataset_size"),
        pas_k=training.get("k", 3),
        pas_context_mode=training.get("context_mode", "full"),
        teacher_model=training.get("teacher") or config.get("model"),
        model_name=config.get("model", ""),
        dataset=training.get("dataset", "math"),
        trace_model=training.get("trace_model", "deepseek/deepseek-r1-distill-qwen-32b"),
        judge=training.get("judge", False),
        judge_model=training.get("judge_model", "deepseek/deepseek-chat"),
    )
    save_dataset(rows, path)
    return path
