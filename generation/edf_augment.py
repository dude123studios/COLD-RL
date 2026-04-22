"""
Explicit Diversity Fine-Tuning (EDF) data synthesis.

Pipeline:
  1. For each problem, generate K alternate solution plans (reuse PAS planning).
  2. Give each plan + problem to a REASONING teacher (e.g. DeepSeek-R1).
     The teacher must produce a full thinking trace that follows the plan.
  3. Training data: input = problem + plan, label = teacher's reasoning trace.

The student learns to execute diverse reasoning strategies on command,
rather than implicitly picking up diversity from seeing varied solutions.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

from generation.rollout import extract_answer
from generation.pas_augment import _parse_plans, _make_client, CACHE_DIR

EDF_CACHE_DIR = CACHE_DIR / "edf"

PLAN_PROMPT = """\
Problem: {problem}

A student solved this correctly using one approach. Identify {k} fundamentally different \
mathematical strategies that could solve this problem.

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

TRACE_PROMPT = """\
Problem: {problem}

You MUST solve this problem using the following specific approach:
{plan}

Think through the problem step by step using ONLY this approach. Show your full \
reasoning process — including exploration, self-checks, and working through the math. \
Do not use any other method. Arrive at the final answer and put it in \\boxed{{}}.

Solve it now."""

JUDGE_PROMPT = """\
You are evaluating whether a math solution follows a specified approach.

Approach specified:
{plan}

Solution trace (first 600 chars):
{trace_excerpt}

Does this solution trace actually follow the specified approach above? \
Answer YES if it clearly uses the described technique/strategy, NO if it uses a different method.

Answer YES or NO only."""


async def _generate_plans_async(
    client: AsyncOpenAI,
    problem_dict: dict,
    k: int,
    model: str,
    semaphore: asyncio.Semaphore,
) -> list[str]:
    """Generate K alternate solution plans for a problem."""
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": PLAN_PROMPT.format(
                            problem=problem_dict["problem"], k=k
                        )}],
                        temperature=0.7,
                        max_tokens=1024,
                    ),
                    timeout=60.0,
                )
                raw = resp.choices[0].message.content
                if raw:
                    return _parse_plans(raw, k)
            except asyncio.TimeoutError:
                if attempt == 2:
                    print(f"[EDF] Planning timed out for {problem_dict['id']}", flush=True)
                else:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == 2:
                    print(f"[EDF] Planning failed for {problem_dict['id']}: {e}", flush=True)
                else:
                    await asyncio.sleep(2 ** attempt)
        return []


async def _generate_trace_async(
    client: AsyncOpenAI,
    problem: str,
    plan: str,
    gold_answer: str,
    model: str,
    semaphore: asyncio.Semaphore,
    timeout: float = 180.0,
) -> Optional[str]:
    """Have the reasoning teacher generate a full trace following the plan."""
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": TRACE_PROMPT.format(
                            problem=problem, plan=plan
                        )}],
                        temperature=0.6,
                        max_tokens=4096,
                    ),
                    timeout=timeout,
                )
                raw = resp.choices[0].message.content
                if raw:
                    predicted = extract_answer(raw)
                    if predicted == gold_answer:
                        return raw.strip()
                return None
            except asyncio.TimeoutError:
                if attempt == 2:
                    print(f"[EDF] Trace timed out after {timeout}s", flush=True)
                else:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == 2:
                    print(f"[EDF] Trace generation failed: {e}", flush=True)
                else:
                    await asyncio.sleep(2 ** attempt)
        return None


async def _judge_trace_async(
    client: AsyncOpenAI,
    plan: str,
    trace: str,
    judge_model: str,
    semaphore: asyncio.Semaphore,
) -> bool:
    """Return True if the trace actually follows the specified plan (per LLM judge)."""
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    plan=plan,
                    trace_excerpt=trace[:600],
                )}],
                temperature=0.0,
                max_tokens=5,
            )
            answer = (resp.choices[0].message.content or "").strip().upper()
            return answer.startswith("YES")
        except Exception as e:
            print(f"[EDF] Judge failed: {e}")
            return True  # fail-open: keep the trace if judge errors


async def _process_problem_async(
    client: AsyncOpenAI,
    problem_dict: dict,
    k: int,
    plan_model: str,
    trace_model: str,
    plan_semaphore: asyncio.Semaphore,
    trace_semaphore: asyncio.Semaphore,
    judge: bool = False,
    judge_model: str = "deepseek/deepseek-chat",
    judge_semaphore: Optional[asyncio.Semaphore] = None,
) -> list[dict]:
    """
    Full EDF pipeline for one problem:
      1. Generate K plans (using plan_model)
      2. Generate all K traces in parallel (using trace_model)
      3. Optionally judge each trace for plan adherence (if judge=True)
      4. Return training rows: input=problem+plan, output=trace
    """
    plans = await _generate_plans_async(client, problem_dict, k, plan_model, plan_semaphore)
    if not plans:
        return []

    # Fire all K trace requests in parallel
    trace_coros = [
        _generate_trace_async(
            client, problem_dict["problem"], plan,
            problem_dict["answer"], trace_model, trace_semaphore,
        )
        for plan in plans
    ]
    traces = await asyncio.gather(*trace_coros)

    results = []
    for i, (plan, trace) in enumerate(zip(plans, traces)):
        if not trace:
            continue

        if judge:
            sem = judge_semaphore or trace_semaphore
            follows_plan = await _judge_trace_async(client, plan, trace, judge_model, sem)
            if not follows_plan:
                continue

        results.append({
            "id": problem_dict["id"],
            "problem": problem_dict["problem"],
            "plan": plan,
            "trace": trace,
            "answer": problem_dict["answer"],
            "source": f"edf_k{i + 1}",
            "judge_verified": judge,
        })
    return results


def build_edf_dataset(
    problems: list[dict],
    k: int = 3,
    plan_model: str = "deepseek/deepseek-chat",
    trace_model: str = "deepseek/deepseek-r1-distill-qwen-32b",
    experiment_id: str = "edf_default",
    max_concurrent: int = 32,
    max_trace_concurrent: int = 32,
    judge: bool = False,
    judge_model: str = "deepseek/deepseek-chat",
    max_judge_concurrent: int = 32,
) -> list[dict]:
    """
    Build EDF training dataset.

    Args:
        problems: list of {id, problem, solution, answer}
        k: number of alternate plans per problem
        plan_model: model for generating plans (fast, non-reasoning)
        trace_model: REASONING model for generating traces (must think step-by-step)
        experiment_id: for caching
        max_concurrent: concurrency limit for plan generation (DeepSeek Chat)
        max_trace_concurrent: concurrency limit for trace generation (DeepSeek R1)
        judge: if True, filter traces that don't follow their specified plan
        judge_model: model used for plan-adherence judging (should be fast/cheap)
        max_judge_concurrent: concurrency limit for judge calls

    Returns:
        list of training rows with {id, problem, plan, trace, answer, source}
    """
    EDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = EDF_CACHE_DIR / f"{experiment_id}.jsonl"

    cached_ids: set[str] = set()
    cached_rows: list[dict] = []
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                row = json.loads(line)
                cached_ids.add(row["id"] + row.get("source", ""))
                cached_rows.append(row)

    missing = [p for p in problems if p["id"] not in {r["id"] for r in cached_rows}]

    if not missing:
        print(f"[EDF] All {len(problems)} problems cached ({len(cached_rows)} rows)")
        return cached_rows

    print(f"[EDF] Generating traces for {len(missing)} problems "
          f"(plan={plan_model}, trace={trace_model}, k={k})")

    if judge:
        print(f"[EDF] Plan-adherence judging ENABLED (judge={judge_model})")
    new_rows = asyncio.run(
        _augment_all(
            missing, k, plan_model, trace_model,
            max_concurrent, max_trace_concurrent,
            judge, judge_model, max_judge_concurrent,
            cache_path,
        )
    )
    all_rows = cached_rows + new_rows
    print(f"[EDF] Total: {len(all_rows)} training examples "
          f"({len(all_rows)/len(problems):.1f} per problem)")
    return all_rows


async def _augment_all(
    problems: list[dict],
    k: int,
    plan_model: str,
    trace_model: str,
    max_concurrent: int,
    max_trace_concurrent: int,
    judge: bool,
    judge_model: str,
    max_judge_concurrent: int,
    cache_path: Path,
) -> list[dict]:
    client = _make_client()
    plan_semaphore = asyncio.Semaphore(max_concurrent)
    trace_semaphore = asyncio.Semaphore(max_trace_concurrent)
    judge_semaphore = asyncio.Semaphore(max_judge_concurrent) if judge else None

    tasks = [
        _process_problem_async(
            client, p, k, plan_model, trace_model,
            plan_semaphore, trace_semaphore,
            judge=judge, judge_model=judge_model, judge_semaphore=judge_semaphore,
        )
        for p in problems
    ]

    all_results = []
    completed = 0
    consecutive_empty = 0
    cache_file = open(cache_path, "a")
    try:
        for coro in asyncio.as_completed(tasks):
            rows = await coro
            all_results.extend(rows)
            for row in rows:
                cache_file.write(json.dumps(row) + "\n")
            cache_file.flush()
            completed += 1

            # Detect credit exhaustion early: if the first 20 problems all produce
            # 0 traces, something is systemically wrong (likely 402). Abort loudly.
            if not rows:
                consecutive_empty += 1
            else:
                consecutive_empty = 0
            if consecutive_empty >= 20 and completed <= 50:
                raise RuntimeError(
                    f"[EDF] ABORT: first {consecutive_empty} problems all produced 0 traces. "
                    f"Check API credits or model availability."
                )

            if completed % 50 == 0:
                print(f"[EDF] {completed}/{len(problems)} problems done, "
                      f"{len(all_results)} traces generated", flush=True)
    finally:
        cache_file.close()

    return all_results


def edf_rows_to_sft_format(rows: list[dict], model_name: str = "") -> list[dict]:
    """
    Convert EDF rows to SFT training format.

    Input to student:  problem + plan (as a system-level instruction)
    Label:             teacher's reasoning trace

    At inference time, we DON'T give the plan — we just evaluate normally.
    The hypothesis is that training on plan-conditioned traces teaches the
    student to naturally explore diverse strategies.
    """
    from generation.build_sft_dataset import _chat_prompt, _make_row

    sft_rows = []
    for row in rows:
        prompt_with_plan = (
            f"{row['problem']}\n\n"
            f"[Approach: {row['plan']}]"
        )
        sft_rows.append(_make_row(
            prompt_with_plan,
            row["trace"],
            row["id"],
            row["source"],
            f"edf_{row.get('source', 'unknown')}",
            model_name,
        ))
    return sft_rows
