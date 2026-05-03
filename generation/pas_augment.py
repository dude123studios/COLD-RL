"""
PAS (Path-Augmented SFT) data synthesis.

The teacher identifies branch points in the original solution — specific steps
where the solver could have taken a different mathematical path — and produces
augmented traces that share a prefix with the original up to that branch point,
then diverge. This teaches the student that at a given reasoning state, multiple
continuations are valid.

Step extraction is only used at eval time (in evaluation/compute_rdiv.py)
to measure diversity of messy model rollouts.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

from generation.rollout import extract_answer

CACHE_DIR = Path(__file__).parent.parent / "results" / "pas_cache"

# Teacher model via OpenRouter
TEACHER_MODEL = "deepseek/deepseek-r1-distill-qwen-32b"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _make_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=api_key or os.environ["OPENROUTER_API_KEY"],
        base_url=base_url or OPENROUTER_BASE,
        timeout=120.0,
    )


TEACHER_PROMPT = """\
Problem: {problem}

Here is one correct solution:
{solution}

Generate {k} alternative solutions that use fundamentally different \
mathematical approaches or strategies from the one above. Each \
alternative must:
- Reach the correct answer
- Use a distinctly different core method (not just different algebra)
- Be self-contained and complete

Format your response exactly as:

## Alternative 1
<full solution here>

## Alternative 2
<full solution here>

(and so on)"""

# For the context-window ablation (replaces full solution with just the first sentence)
TEACHER_PROMPT_FIRST_SENTENCE = """\
Problem: {problem}

Here is the beginning of one correct solution:
{solution_prefix}

Generate {k} alternative solutions that use fundamentally different \
mathematical approaches or strategies. Each alternative must:
- Reach the correct answer
- Use a distinctly different core method
- Be self-contained and complete

## Alternative 1
"""

# Blind alternative generation: teacher gets no original solution, generates from scratch
TEACHER_PROMPT_BLIND = """\
Problem: {problem}

Generate {k} complete, correct solutions to this problem using fundamentally different \
mathematical approaches or strategies. Each solution must:
- Reach the correct answer
- Use a distinctly different core method (not just different algebra)
- Be self-contained and complete

Format your response exactly as:

## Alternative 1
<full solution here>

## Alternative 2
<full solution here>

(and so on)"""

# Style-preserving full rewrite: completely different mathematical approach,
# but rewrites so it sounds like the same solver wrote it (same voice, notation, LaTeX style).
# This is the core PAS variant that actually works — on-policy style + approach diversity.
TEACHER_PROMPT_STYLE_REWRITE = """\
Problem: {problem}

Here is a solution written by a student:
{solution}

Write {k} alternative solution(s) to the same problem. Each alternative must:
- Use a COMPLETELY DIFFERENT mathematical strategy or approach (not just different algebra steps)
- Be written in the same voice, level of detail, and notation style as the student's solution above
- Reach the same correct final answer
- Read as if the same student wrote it using a different method

The goal is that each alternative feels stylistically identical to the original, \
but takes an entirely different mathematical path to the answer.

Format your response exactly as:

## Alternative 1
<full solution here>

## Alternative 2
<full solution here>

(and so on)"""

# Path-Augmented SFT: teacher identifies branch points in the original trace and
# produces augmented traces that share the original prefix up to that branch point,
# then take a different mathematical path to the same answer.
# Input is always a model-generated solution (from rejection sampling), never textbook LaTeX.
TEACHER_PROMPT_MINIMAL_EDIT = """\
Problem: {problem}

Solution trace:
{solution}

Identify {k} branch point(s) in this solution — specific steps where the solver \
could have chosen a different mathematical technique, formula, or approach and still \
reached the correct answer.

For each branch point, write the full modified solution that:
- Keeps all steps BEFORE the branch point word-for-word identical
- Takes the alternative mathematical path from that point onward
- Reaches the same correct final answer

Format your response exactly as:

## Alternative 1
Branch point: <quote the exact sentence or step where the solution diverges>
Modified solution:
<full solution: identical prefix up to branch point, then alternative path to answer>

## Alternative 2
Branch point: <quote the exact sentence or step where the solution diverges>
Modified solution:
<full solution: identical prefix up to branch point, then alternative path to answer>

(and so on)"""


TEACHER_PROMPT_TRACE_PLAN = """\
Problem: {problem}

A student produced the following solution trace:
{trace}

The student used one particular approach. Identify {k} fundamentally different mathematical \
strategies that could solve this same problem instead.

For each alternative strategy, provide:
- The core mathematical technique or insight (1-2 sentences)
- Key steps in this approach (brief outline, not a full solution)

Format exactly as:

## Plan 1
Core: [the key mathematical technique]
Steps: [brief step outline]

## Plan 2
Core: [the key mathematical technique]
Steps: [brief step outline]

(and so on)"""


TEACHER_PROMPT_TRACE_EXPAND = """\
Problem: {problem}

Here is a student's original solution trace:
{trace}

Alternative approach to follow instead:
{plan}

Rewrite the solution trace so the student naturally discovers and works through this \
alternative approach. Requirements:
- Match the original trace's writing style, voice, notation, and level of detail exactly
- Match the approximate length of the original trace
- Show the student working through the math naturally — including any exploratory steps, \
self-checks, or minor corrections that fit the style
- Follow the alternative approach throughout (do not fall back to the original method)
- Reach the same correct final answer
- End with the answer in the same format as the original (e.g. \\boxed{{}})

Write only the rewritten trace, nothing else."""


def _parse_plans(text: str, k: int) -> list[str]:
    """Parse ## Plan N sections from teacher planning response."""
    parts = []
    for i in range(1, k + 1):
        start_marker = f"## Plan {i}"
        end_marker = f"## Plan {i + 1}" if i < k else None
        start = text.find(start_marker)
        if start == -1:
            continue
        start += len(start_marker)
        if end_marker:
            end = text.find(end_marker)
            block = text[start:end].strip() if end != -1 else text[start:].strip()
        else:
            block = text[start:].strip()
        if block:
            parts.append(block)
    # Fallback: split on double newline sections
    if not parts and text.strip():
        chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 40]
        parts = chunks[:k]
    return parts


async def _plan_and_expand_problem_async(
    client: AsyncOpenAI,
    problem_dict: dict,
    k: int,
    model: str,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """
    K+1 call pipeline:
      Call 1: generate K alternative approach plans from the original trace
      Calls 2..K+1: expand each plan into a full trace matching the original style/length
    Returns list of {id, problem, solution, answer, source} dicts with correct answers.
    """
    async with semaphore:
        problem = problem_dict["problem"]
        trace = problem_dict["solution"]
        gold_answer = problem_dict["answer"]

        # Call 1: planning
        plans = []
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": TEACHER_PROMPT_TRACE_PLAN.format(
                        problem=problem, trace=trace, k=k
                    )}],
                    temperature=0.7,
                    max_tokens=1024,
                )
                raw = resp.choices[0].message.content
                if raw:
                    plans = _parse_plans(raw, k)
                    break
            except Exception as e:
                if attempt == 2:
                    print(f"[PAS-trace] Planning call failed: {e}")
                else:
                    await asyncio.sleep(2 ** attempt)

        if not plans:
            return []

        # Calls 2..K+1: expand each plan into a trace
        async def expand_one(plan: str, idx: int) -> dict | None:
            for attempt in range(3):
                try:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": TEACHER_PROMPT_TRACE_EXPAND.format(
                            problem=problem, trace=trace, plan=plan
                        )}],
                        temperature=0.8,
                        max_tokens=max(len(trace.split()) * 3, 2048),
                    )
                    raw = resp.choices[0].message.content
                    if raw:
                        predicted = extract_answer(raw)
                        if predicted == gold_answer:
                            return {
                                "id": problem_dict["id"],
                                "problem": problem,
                                "solution": raw.strip(),
                                "answer": gold_answer,
                                "source": f"pas_k{idx + 1}",
                            }
                        return None  # wrong answer, discard
                except Exception as e:
                    if attempt == 2:
                        print(f"[PAS-trace] Expand call failed (plan {idx+1}): {e}")
                    else:
                        await asyncio.sleep(2 ** attempt)
            return None

        expand_tasks = [expand_one(plan, i) for i, plan in enumerate(plans)]
        expanded = await asyncio.gather(*expand_tasks)
        return [r for r in expanded if r is not None]


def _parse_alternatives(text: str, k: int) -> list[str]:
    """
    Parse ## Alternative N sections from teacher response.

    For the branch-point format, each section contains:
        Branch point: <quoted step>
        Modified solution:
        <full solution text>

    We extract only the Modified solution block. Falls back to taking the
    full section content if the sub-structure is missing (e.g. for the
    TEACHER_PROMPT / TEACHER_PROMPT_FIRST_SENTENCE formats).
    """
    parts = []
    for i in range(1, k + 1):
        start_marker = f"## Alternative {i}"
        end_marker = f"## Alternative {i + 1}" if i < k else None

        start = text.find(start_marker)
        if start == -1:
            continue
        start += len(start_marker)

        if end_marker:
            end = text.find(end_marker)
            block = text[start:end].strip() if end != -1 else text[start:].strip()
        else:
            block = text[start:].strip()

        if not block:
            continue

        # If this is branch-point format, extract only the Modified solution
        mod_marker = "Modified solution:"
        mod_idx = block.find(mod_marker)
        if mod_idx != -1:
            content = block[mod_idx + len(mod_marker):].strip()
        else:
            # Plain format (TEACHER_PROMPT / TEACHER_PROMPT_FIRST_SENTENCE)
            content = block

        if content:
            parts.append(content)

    # Fallback: if parsing failed entirely, split on double newlines
    if not parts and text.strip():
        chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 100]
        parts = chunks[:k]

    return parts


async def _generate_alternatives_async(
    client: AsyncOpenAI,
    problem: str,
    solution: str,
    k: int,
    model: str,
    context_mode: str = "full",  # "full", "first_sentence", "minimal_edit", "style_rewrite", "blind"
) -> list[str]:
    if context_mode == "first_sentence":
        first_sentence = solution.split(".")[0] + "." if "." in solution else solution[:100]
        prompt = TEACHER_PROMPT_FIRST_SENTENCE.format(
            problem=problem,
            solution_prefix=first_sentence,
            k=k,
        )
    elif context_mode == "minimal_edit":
        prompt = TEACHER_PROMPT_MINIMAL_EDIT.format(problem=problem, solution=solution, k=k)
    elif context_mode == "style_rewrite":
        prompt = TEACHER_PROMPT_STYLE_REWRITE.format(problem=problem, solution=solution, k=k)
    elif context_mode == "blind":
        prompt = TEACHER_PROMPT_BLIND.format(problem=problem, k=k)
    else:
        prompt = TEACHER_PROMPT.format(problem=problem, solution=solution, k=k)

    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=4096,
            )
            raw = resp.choices[0].message.content
            if raw is None:
                if attempt == 2:
                    print(f"[PAS] API returned None content after 3 attempts (content filter?)")
                    return []
                await asyncio.sleep(2 ** attempt)
                continue
            return _parse_alternatives(raw, k)
        except Exception as e:
            err = str(e)
            if "401" in err or "authentication" in err.lower() or "missing" in err.lower():
                raise RuntimeError(
                    f"[PAS] Auth failed — set OPENROUTER_API_KEY before running: {e}"
                ) from e
            if attempt == 2:
                print(f"[PAS] Teacher API failed after 3 attempts: {e}")
                return []
            await asyncio.sleep(2 ** attempt)

    return []


async def _augment_problem_async(
    client: AsyncOpenAI,
    problem_dict: dict,
    k: int,
    model: str,
    context_mode: str,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """
    Generate k alternative solutions for one problem.
    Returns list of {problem, solution, answer, source} dicts.
    Filters out alternatives with wrong answers.
    """
    if context_mode == "trace_expand":
        return await _plan_and_expand_problem_async(client, problem_dict, k, model, semaphore)

    async with semaphore:
        problem = problem_dict["problem"]
        solution = problem_dict["solution"]
        gold_answer = problem_dict["answer"]

        alternatives = await _generate_alternatives_async(
            client, problem, solution, k, model, context_mode
        )

        results = []
        for i, alt in enumerate(alternatives):
            predicted = extract_answer(alt)
            if predicted == gold_answer:
                results.append(
                    {
                        "id": problem_dict["id"],
                        "problem": problem,
                        "solution": alt,
                        "answer": gold_answer,
                        "source": f"pas_k{i+1}",
                    }
                )
            else:
                # Keep but mark as filtered — for debugging only, not used in training
                pass

        return results


def _augment_dataset_local(
    problems: list[dict],
    model_name: str,
    k: int,
    context_mode: str,
    experiment_id: str,
) -> list[dict]:
    """
    Generate PAS alternatives using a local vLLM model (no API dependency).
    Batches all problems into one vLLM call for efficiency.
    """
    from vllm import LLM, SamplingParams

    cache_path = CACHE_DIR / f"{experiment_id}.jsonl"
    cached_ids: set[str] = set()
    cached_rows: list[dict] = []
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                row = json.loads(line)
                cached_ids.add(row["id"])
                cached_rows.append(row)

    missing = [p for p in problems if p["id"] not in cached_ids]
    if not missing:
        print(f"[PAS-local] All {len(problems)} problems cached")
        return cached_rows

    print(f"[PAS-local] Generating alternatives for {len(missing)} problems with {model_name}")

    # Build prompts
    prompts = []
    for p in missing:
        sol = p["solution"]
        if context_mode == "minimal_edit":
            user_msg = TEACHER_PROMPT_MINIMAL_EDIT.format(
                problem=p["problem"], solution=sol, k=k
            )
        elif context_mode == "first_sentence":
            first = sol.split(".")[0] + "." if "." in sol else sol[:100]
            user_msg = TEACHER_PROMPT_FIRST_SENTENCE.format(
                problem=p["problem"], solution_prefix=first, k=k
            )
        elif context_mode == "style_rewrite":
            user_msg = TEACHER_PROMPT_STYLE_REWRITE.format(
                problem=p["problem"], solution=sol, k=k
            )
        elif context_mode == "blind":
            user_msg = TEACHER_PROMPT_BLIND.format(problem=p["problem"], k=k)
        else:
            user_msg = TEACHER_PROMPT.format(problem=p["problem"], solution=sol, k=k)

        # Wrap in model-appropriate chat format
        from evaluation.vllm_generate import _format_chat
        prompts.append(_format_chat(user_msg, model_name))

    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.40,
        enforce_eager=True,
        disable_log_stats=True,
    )
    sampling = SamplingParams(temperature=0.8, max_tokens=4096)
    outputs = llm.generate(prompts, sampling)
    del llm

    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    new_rows = []
    with open(cache_path, "a") as f:
        for p, out in zip(missing, outputs):
            raw = out.outputs[0].text
            alts = _parse_alternatives(raw, k)
            for i, alt in enumerate(alts):
                predicted = extract_answer(alt)
                if predicted == p["answer"]:
                    row = {
                        "id": p["id"],
                        "problem": p["problem"],
                        "solution": alt,
                        "answer": p["answer"],
                        "source": f"pas_k{i+1}",
                    }
                    new_rows.append(row)
                    f.write(json.dumps(row) + "\n")

    n_alts = len(new_rows)
    print(f"[PAS-local] Generated {n_alts} valid alternatives from {len(missing)} problems "
          f"({n_alts/len(missing):.1f} per problem)")
    return cached_rows + new_rows


class PASAugmenter:
    def __init__(
        self,
        teacher_model: str = TEACHER_MODEL,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_concurrent: int = 8,
        use_local: bool = False,
    ):
        self.teacher_model = teacher_model
        self.api_key = api_key
        self.api_base = api_base
        self.max_concurrent = max_concurrent
        self.use_local = use_local
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, experiment_id: str) -> Path:
        return CACHE_DIR / f"{experiment_id}.jsonl"

    def augment_dataset(
        self,
        problems: list[dict],
        k: int = 3,
        context_mode: str = "full",
        experiment_id: str = "default",
        include_original: bool = True,
        filtering_threshold: float = 0.0,
    ) -> list[dict]:
        """
        Augment all problems with PAS alternatives.

        Args:
            problems: list of {id, problem, solution, answer}
            k: number of alternative solutions per problem
            context_mode: "full" (give full original solution) or
                          "first_sentence" (ablation: minimal context)
            experiment_id: used for caching
            include_original: whether to include the original solutions too

        Returns:
            list of training examples
        """
        if self.use_local:
            all_rows = _augment_dataset_local(
                problems, self.teacher_model, k, context_mode, experiment_id
            )
            result = []
            if include_original:
                for p in problems:
                    result.append({
                        "id": p["id"], "problem": p["problem"],
                        "solution": p["solution"], "answer": p["answer"],
                        "source": "original",
                    })
            result.extend(all_rows)
            return result

        cache_path = self._cache_path(experiment_id)

        # Load cached
        cached_ids: set[str] = set()
        cached_rows: list[dict] = []
        if cache_path.exists():
            with open(cache_path) as f:
                for line in f:
                    row = json.loads(line)
                    cached_ids.add(row["id"] + row.get("source", ""))
                    cached_rows.append(row)

        missing = [p for p in problems if p["id"] not in {r["id"] for r in cached_rows if r["source"] != "original"}]

        if missing:
            new_rows = asyncio.run(
                self._augment_all(missing, k, context_mode, cache_path)
            )
            cached_rows.extend(new_rows)

        # Build final dataset
        result = []
        if include_original:
            for p in problems:
                result.append(
                    {
                        "id": p["id"],
                        "problem": p["problem"],
                        "solution": p["solution"],
                        "answer": p["answer"],
                        "source": "original",
                    }
                )

        # Add PAS alternatives (from cache)
        pas_by_id: dict[str, list[dict]] = {}
        for row in cached_rows:
            if row.get("source", "").startswith("pas_"):
                pas_by_id.setdefault(row["id"], []).append(row)

        if filtering_threshold > 0.0:
            # Filter out problems where fraction of correct alternatives < threshold
            n_filtered = 0
            for prob_id, rows in pas_by_id.items():
                frac = len(rows) / k
                if frac >= filtering_threshold:
                    result.extend(rows)
                else:
                    n_filtered += 1
            if n_filtered:
                print(f"[PAS] Filtered {n_filtered} problems below τ={filtering_threshold:.2f}")
        else:
            for rows in pas_by_id.values():
                result.extend(rows)

        return result

    async def _augment_all(
        self,
        problems: list[dict],
        k: int,
        context_mode: str,
        cache_path: Path,
    ) -> list[dict]:
        client = _make_client(self.api_key, self.api_base)
        semaphore = asyncio.Semaphore(self.max_concurrent)

        tasks = [
            _augment_problem_async(
                client, p, k, self.teacher_model, context_mode, semaphore
            )
            for p in problems
        ]

        all_results = []
        completed = 0
        cache_file = open(cache_path, "a")
        try:
            for coro in asyncio.as_completed(tasks):
                rows = await coro
                all_results.extend(rows)
                for row in rows:
                    cache_file.write(json.dumps(row) + "\n")
                cache_file.flush()
                completed += 1
                if completed % 100 == 0:
                    print(f"[PAS] {completed}/{len(problems)} problems augmented, "
                          f"{len(all_results)} alternatives generated", flush=True)
        finally:
            cache_file.close()

        return all_results
