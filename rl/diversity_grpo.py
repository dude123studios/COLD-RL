"""
Diversity-GRPO Training

Differences from standard GRPO / DARLING:
  1. Explicit diversity prompting: each of N rollouts is prompted with a
     different method number X (1..N), asking for "solution approach X".
  2. Step-level diversity reward (not cluster-level like DARLING):
     diversity_score_i = max cosine distance between any step in trace i
     and any step in any other trace.
  3. Gated combined reward: r_i = r_correct + lambda * r_correct * r_diversity.

Hyperparameters follow DARLING (arXiv:2509.02534):
  - N = 8 rollouts per problem (one per method number)
  - LR = 1e-6
  - Global batch size = 256 (prompt-level), effectively 32 problems × 8 rollouts
  - max_new_tokens = 8192
  - clip ε = 0.2
  - KL β = 0.001
  - temperature = 0.8
  - warmup_ratio = 0.1
"""

import gc
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

METHOD_SYSTEM_PROMPT = (
    "You are a brilliant mathematician. You will be shown a math problem and "
    "asked to solve it using a specific solution approach. Think step by step."
)

METHOD_PROMPT_TEMPLATE = (
    "Solve the following problem using mathematical approach #{method_id} of {n_methods} "
    "(each approach should be conceptually distinct — e.g., algebraic, geometric, "
    "number-theoretic, direct computation, etc.).\n\n"
    "Problem: {problem}\n\n"
    "Approach #{method_id}: Solve this step by step and conclude with "
    "\\boxed{{your final answer}}."
)


def format_method_prompt(problem: str, method_id: int, n_methods: int = 8) -> str:
    return METHOD_PROMPT_TEMPLATE.format(
        method_id=method_id,
        n_methods=n_methods,
        problem=problem,
    )


def build_chat_prompt(tokenizer, problem: str, method_id: int, n_methods: int = 8) -> str:
    """Build the full chat-formatted prompt string."""
    messages = [
        {"role": "system", "content": METHOD_SYSTEM_PROMPT},
        {"role": "user", "content": format_method_prompt(problem, method_id, n_methods)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Rollout generation via vLLM
# ---------------------------------------------------------------------------

def _estimate_model_gb(model_name: str) -> float:
    """Rough estimate of model size in bfloat16 based on parameter count in name."""
    name = model_name.lower()
    for tag, gb in [("70b", 140), ("32b", 64), ("14b", 28), ("13b", 26),
                    ("8b", 16), ("7b", 14), ("3b", 6), ("1.5b", 3), ("1b", 2)]:
        if tag in name:
            return gb
    return 14  # default to 7B estimate


def _vllm_colocated_util(slack_gb: float, util_cap: float, util_floor: float) -> tuple[float, float, float]:
    """Return (util, free_gb, total_gb) from driver free memory; caller must sync/empty_cache first."""
    free_b, total_b = torch.cuda.mem_get_info(0)
    total_gb = total_b / 1e9
    free_gb = free_b / 1e9
    util = (free_gb - slack_gb) / total_gb
    util = min(util_cap, max(util_floor, util))
    return util, free_gb, total_gb


def _is_vllm_init_memory_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda out of memory" in msg
        or "memory utilization" in msg
        or "engine core initialization failed" in msg
        or "enginecore failed to start" in msg
    )


def create_vllm_engine(
    model_path: str,
    lora_rank: int = 64,
    use_lora: bool = True,
    max_new_tokens: int = 8192,
):
    """
    Create a vLLM engine on the current CUDA device while the HF training model
    stays resident.  Sizing uses driver free memory (mem_get_info), a conservative
    util cap, and retries with lower utilization when EngineCore hits OOM during init.
    """
    from vllm import LLM

    # vLLM compares driver free to (gpu_memory_utilization * total) at startup, but
    # the worker process can still OOM while loading weights + KV if util is too
    # aggressive for fragmented post-teardown memory — cap util and retry lower.
    slack_gb = 2.5
    util_cap = 0.38
    util_floor = 0.20
    max_attempts = 6
    scale_retry = 0.88

    effective_max = min(max_new_tokens, 4096)
    max_model_len = effective_max + 512
    llm_kwargs = dict(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enable_prefix_caching=not use_lora,
        max_model_len=max_model_len,
        disable_log_stats=True,
    )
    if use_lora:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = lora_rank
        llm_kwargs["enforce_eager"] = True

    last_err: Optional[BaseException] = None
    util = util_cap
    for attempt in range(max_attempts):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        util, free_gb, gpu_total_gb = _vllm_colocated_util(slack_gb, util, util_floor)
        llm_kwargs["gpu_memory_utilization"] = util
        logger.info(
            f"[vllm] Creating engine attempt {attempt + 1}/{max_attempts}  util={util:.3f}  "
            f"(driver_free={free_gb:.1f} GB / {gpu_total_gb:.1f} GB total)"
        )
        try:
            return LLM(**llm_kwargs)
        except (RuntimeError, ValueError) as e:
            last_err = e
            if not _is_vllm_init_memory_error(e):
                raise
            util = max(util_floor, util * scale_retry)
            logger.warning(f"[vllm] init failed ({e}); retrying with util={util:.3f}")
            time.sleep(1.0)

    assert last_err is not None
    raise last_err


def destroy_vllm_engine(llm) -> None:
    """Release vLLM GPU memory so HF training can use the full device."""
    if llm is None:
        return
    try:
        eng = getattr(llm, "llm_engine", None)
        core = getattr(eng, "engine_core", None) if eng is not None else None
        if core is not None and hasattr(core, "shutdown"):
            core.shutdown()
    except Exception as e:
        logger.warning(f"[vllm] engine shutdown failed (continuing): {e}")
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _logsumexp_lastdim_chunked(x: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
    """
    Stable logsumexp on dim=-1 with smaller peak memory than torch.logsumexp
    on huge vocab (7B models ~152k logits): avoids multi-GB fusion buffers.
    """
    if x.shape[-1] <= chunk_size:
        return torch.logsumexp(x, dim=-1)
    m = x.max(dim=-1, keepdim=True).values
    acc = None
    V = x.shape[-1]
    for c in range(0, V, chunk_size):
        sl = x[..., c : c + chunk_size]
        part = (sl - m).exp().sum(dim=-1)
        acc = part if acc is None else acc + part
    return m.squeeze(-1) + acc.log()


def generate_rollouts(
    prompts: list[str],
    model_path: str,
    temperature: float = 0.8,
    max_new_tokens: int = 8192,
    gpu_id: int = 0,
    training_model: torch.nn.Module = None,
    lora_path: Optional[str] = None,
    lora_rank: int = 64,
    top_p: float = 1.0,
    llm=None,
    lora_slot_id: int = 1,
) -> list[str]:
    """
    Generate completions for all prompts using vLLM.

    Persistent mode (llm is not None):
      Pass a pre-created LLM engine -- no CPU offload, no init/teardown overhead.
      Use an incrementing lora_slot_id each call so vLLM re-reads the adapter
      from disk rather than serving stale cached weights from a prior step.

    Ephemeral mode (llm is None, fallback):
      CPU-offload the training model, spin up a fresh LLM, generate, destroy.
    """
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=min(max_new_tokens, 4096),
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    # Unique integer ID per step forces vLLM to reload the adapter from disk
    # rather than reusing stale cached weights from the previous step.
    lora_req = (
        LoRARequest(f"adapter_{lora_slot_id}", lora_slot_id, lora_path)
        if lora_path else None
    )

    if llm is not None:
        # ---- Persistent path: engine already running, just swap the LoRA adapter ----
        outputs = llm.generate(prompts, params, lora_request=lora_req)
        return [out.outputs[0].text for out in outputs]

    # ---- Ephemeral path (fallback) ----
    from vllm import LLM

    gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    model_size_gb = _estimate_model_gb(model_path)

    # CPU-offload training model before vLLM -- vLLM needs most of VRAM for KV cache
    offloaded = False
    if training_model is not None:
        logger.info(f"[gen] Offloading training model to CPU ({model_size_gb:.0f}GB model, "
                    f"{gpu_total_gb:.0f}GB GPU) for vLLM generation")
        training_model.cpu()
        torch.cuda.empty_cache()
        offloaded = True

    util = min(0.82, (gpu_total_gb - 6) / gpu_total_gb)
    logger.info(
        f"[gen] vLLM gpu_memory_utilization={util:.2f}"
        + (f"  LoRA adapter={lora_path}" if lora_path else "")
    )

    effective_max = min(max_new_tokens, 4096)
    max_model_len = effective_max + 512
    llm_kwargs = dict(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=util,
        enable_prefix_caching=not bool(lora_path),
        max_model_len=max_model_len,
        disable_log_stats=True,
    )
    if lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = lora_rank
        llm_kwargs["enforce_eager"] = True
    llm_local = LLM(**llm_kwargs)
    outputs = llm_local.generate(prompts, params, lora_request=lora_req)
    completions = [out.outputs[0].text for out in outputs]
    del llm_local
    torch.cuda.empty_cache()

    if offloaded and training_model is not None:
        logger.info("[gen] Moving training model back to cuda:0")
        training_model.to("cuda:0")

    return completions


# ---------------------------------------------------------------------------
# GRPO advantage computation
# ---------------------------------------------------------------------------

def compute_grpo_advantages(
    rewards: list[float],
    n_rollouts: int,
    eps: float = 1e-8,
) -> list[float]:
    """
    Standard GRPO: normalize rewards within each group of n_rollouts.
    rewards is a flat list [g0_r0, g0_r1, ..., g0_rN, g1_r0, ...]
    Returns advantages in same order.
    """
    advantages = []
    n_groups = len(rewards) // n_rollouts
    for g in range(n_groups):
        group = rewards[g * n_rollouts: (g + 1) * n_rollouts]
        mean_r = np.mean(group)
        std_r = np.std(group) + eps
        for r in group:
            advantages.append((r - mean_r) / std_r)
    return advantages


# ---------------------------------------------------------------------------
# Per-token log-probability computation
# ---------------------------------------------------------------------------

def compute_log_probs(
    model: AutoModelForCausalLM,
    tokenizer,
    prompts: list[str],
    completions: list[str],
    device: str = "cuda",
) -> list[torch.Tensor]:
    """
    Returns per-token log-probs for each (prompt+completion) sequence,
    masked to completion tokens only.
    """
    log_probs_list = []
    for prompt, completion in zip(prompts, completions):
        full = prompt + completion
        enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=8192)
        prompt_len = len(tokenizer(prompt, return_tensors="pt").input_ids[0])

        input_ids = enc.input_ids.to(device)
        with torch.no_grad():
            out = model(input_ids)
        logits = out.logits  # (1, seq_len, vocab)

        # Shift: logits[t] predicts token[t+1]
        logits = logits[0, :-1, :]  # (seq-1, vocab)
        targets = input_ids[0, 1:]  # (seq-1,)

        log_p = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_p[torch.arange(len(targets)), targets]

        # Mask to completion tokens only
        comp_log_probs = token_log_probs[prompt_len - 1:]
        log_probs_list.append(comp_log_probs)
    return log_probs_list


def compute_ref_log_probs_batched(
    model,
    tokenizer,
    prompts: list[str],
    completions: list[str],
    device: str = "cuda",
    ref_batch_size: int = 32,
) -> list[torch.Tensor]:
    """
    Compute reference log-probs in large batches (no grad => no mini-batch needed).

    The original per-sequence loop ran 256 sequential forward passes with Python
    overhead (~600 ms/seq => ~154 s total).  Batching to ref_batch_size=32 means
    8 batched forward passes where the GPU processes 32 seqs in parallel, amortising
    weight-loading and Python overhead (~3-5 s total, ~40-50x faster).

    Sequences are sorted by length to minimise right-padding waste.  Right-padding
    is safe for causal attention: actual tokens only attend left, so log-probs at
    real positions are identical to unbatched results.
    """
    n = len(prompts)
    results: list = [None] * n

    seqs = []
    for i, (prompt, comp) in enumerate(zip(prompts, completions)):
        full_ids = tokenizer(
            prompt + comp, return_tensors="pt", truncation=True, max_length=8192
        ).input_ids[0]
        prompt_len = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
        seqs.append((i, full_ids, prompt_len))

    seqs.sort(key=lambda x: len(x[1]))

    with torch.no_grad():
        for batch_start in range(0, n, ref_batch_size):
            batch    = seqs[batch_start: batch_start + ref_batch_size]
            idxs     = [item[0] for item in batch]
            ids_list = [item[1] for item in batch]
            plens    = [item[2] for item in batch]
            max_len  = max(len(ids) for ids in ids_list)

            padded = torch.full(
                (len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long
            )
            attn = torch.zeros(len(batch), max_len, dtype=torch.long)
            for j, ids in enumerate(ids_list):
                padded[j, :len(ids)] = ids
                attn[j,   :len(ids)] = 1

            logits = model(
                input_ids=padded.to(device),
                attention_mask=attn.to(device),
            ).logits  # (B, L, V)

            # Memory-efficient log-prob: logsumexp trick avoids materialising the
            # full (B, L-1, V) log_softmax tensor (~6 GB for batch=32 at vocab=152K).
            # Instead: log_p[t] = logits[t] - logsumexp(logits[t, :])
            # Each intermediate is (B, L-1) rather than (B, L-1, V).
            targets = padded[:, 1:].to(device)                                 # (B, L-1)
            shifted = logits[:, :-1, :]                                         # view
            target_logits = shifted.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
            log_sum_exp   = _logsumexp_lastdim_chunked(shifted, chunk_size=8192)
            token_lp      = (target_logits - log_sum_exp)                       # (B, L-1)
            del logits, shifted, target_logits, log_sum_exp                     # free VRAM

            for j, (orig_idx, ids, plen) in enumerate(zip(idxs, ids_list, plens)):
                seq_len = len(ids) - 1
                results[orig_idx] = token_lp[j, plen - 1 : seq_len].cpu()

    return results


def grpo_loss_and_backward(
    model: AutoModelForCausalLM,
    tokenizer,
    prompts: list[str],
    completions: list[str],
    advantages: list[float],
    ref_log_probs: list[torch.Tensor],
    clip_eps: float = 0.2,
    kl_beta: float = 0.001,
    device: str = "cuda",
    grad_accum: int = 1,
    mini_batch: int = 16,
) -> float:
    """
    Compute GRPO loss and immediately call backward() on each mini-batch.

    Gradient note: the gradient is computed ONLY over the completion tokens.
    The prompt tokens (which include the "Approach #X" conditioning) are included
    in the forward pass for attention context but are masked out of the loss
    via the [prompt_len - 1:] slice.  This ensures the policy update teaches the
    model *how* to answer a given approach, not *which* approach to pick.

    Memory note: accumulating ALL 256 computation graphs before backward() causes
    OOM on 48GB GPUs with long completions.  Instead, we process `mini_batch`
    completions at a time and immediately call backward(), accumulating the
    weight gradients rather than the activation graphs.
    """
    total_loss_val = 0.0
    count = 0
    n = len(prompts)

    for batch_start in range(0, n, mini_batch):
        batch_end = min(batch_start + mini_batch, n)
        batch_losses: list[torch.Tensor] = []

        for prompt, completion, adv, ref_lp in zip(
            prompts[batch_start:batch_end],
            completions[batch_start:batch_end],
            advantages[batch_start:batch_end],
            ref_log_probs[batch_start:batch_end],
        ):
            if not completion.strip():
                continue

            full = prompt + completion
            enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=8192)
            prompt_len = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
            input_ids = enc.input_ids.to(device)

            out = model(input_ids)
            logits = out.logits[0, :-1, :]
            targets = input_ids[0, 1:]
            log_p = torch.nn.functional.log_softmax(logits, dim=-1)
            # Gradient only over completion tokens (prompt tokens masked out)
            curr_log_probs = log_p[torch.arange(len(targets)), targets][prompt_len - 1:]

            T = min(len(curr_log_probs), len(ref_lp))
            if T == 0:
                continue

            curr = curr_log_probs[:T]
            ref = ref_lp[:T].to(device)

            log_ratio = curr - ref.detach()
            ratio = log_ratio.exp()

            adv_t = torch.tensor(adv, device=device)
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            policy_loss = -torch.min(ratio * adv_t, clipped * adv_t).mean()

            # Per-token KL: ref_log_prob - curr_log_prob (expected under ref)
            kl = (ref.detach().exp() * (ref.detach() - curr)).mean()

            batch_losses.append(policy_loss + kl_beta * kl)
            count += 1

        if not batch_losses:
            continue

        # Backward immediately — only this mini-batch's activations are in memory
        mb_loss = torch.stack(batch_losses).mean()
        total_loss_val += float(mb_loss.item())
        # Scale by mini_batch / n so that each sample contributes equally,
        # and by 1/grad_accum for the outer gradient accumulation.
        scale = (len(batch_losses) / max(count, 1)) / grad_accum
        (mb_loss * scale).backward()

    if count == 0:
        # No valid completions at all — emit a zero-grad dummy step
        dummy = next(p for p in model.parameters() if p.requires_grad)
        (dummy.sum() * 0.0 / grad_accum).backward()

    return total_loss_val / max(count // mini_batch, 1)


# ---------------------------------------------------------------------------
# pass@k evaluation  (COLD methodology: approaches #1→#8 × 2 = 16 samples)
# ---------------------------------------------------------------------------

def _pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator from Chen et al. (HumanEval):
      pass@k = 1 - C(n-c, k) / C(n, k)
    n = total samples, c = correct samples, k = samples to report.
    """
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod([(n - c - i) / (n - i) for i in range(k)]))


def eval_pass_at_k(
    cfg: "DiversityGRPOConfig",
    tokenizer,
    lora_adapter_dir: str,
    problems: list[dict],
    n_eval: int = 50,
    approach_cycles: int = 2,   # 8 approaches x 2 = 16 samples per problem
    gpu_id: int = 0,
    training_model: Optional[torch.nn.Module] = None,
    llm=None,
    lora_slot_id: int = 1,
) -> dict:
    """
    Evaluate pass@k (k=1,4,8,16) using the COLD methodology:
      - For each problem, generate 'n_methods × approach_cycles' completions
        by cycling through approach prompts #1 → #n_methods, approach_cycles times.
      - This lets the model express different methodologies, measuring both
        accuracy (pass@1) and coverage across diverse strategies (pass@16).

    Only uses vLLM for generation (no gradient computation).
    Returns a dict with pass@1, pass@4, pass@8, pass@16, n_problems, n_correct.
    """
    from rl.diversity_reward import is_correct as _is_correct

    eval_probs = random.sample(problems, min(n_eval, len(problems)))
    n_methods = cfg.n_rollouts  # typically 8
    n_samples = n_methods * approach_cycles  # 16

    # Build all prompts: cycle approaches 1..n_methods × approach_cycles
    flat_prompts: list[str] = []
    flat_gold: list[str] = []
    for prob in eval_probs:
        for cycle in range(approach_cycles):
            for method_id in range(1, n_methods + 1):
                flat_prompts.append(
                    build_chat_prompt(tokenizer, prob["problem"], method_id, n_methods)
                )
                flat_gold.append(prob["answer"])

    completions = generate_rollouts(
        flat_prompts,
        model_path=cfg.model_name,
        lora_path=lora_adapter_dir,
        lora_rank=cfg.lora_r,
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
        gpu_id=gpu_id,
        training_model=training_model if llm is None else None,
        llm=llm,
        lora_slot_id=lora_slot_id,
    )

    results = {"pass@1": 0.0, "pass@4": 0.0, "pass@8": 0.0, "pass@16": 0.0,
               "n_problems": len(eval_probs), "n_samples": n_samples}

    p1s, p4s, p8s, p16s = [], [], [], []
    for pidx in range(len(eval_probs)):
        start = pidx * n_samples
        group_completions = completions[start: start + n_samples]
        group_gold = flat_gold[start]
        c = sum(_is_correct(comp, group_gold, cfg.benchmark) for comp in group_completions)
        n = n_samples
        p1s.append(_pass_at_k(n, c, 1))
        p4s.append(_pass_at_k(n, c, 4))
        p8s.append(_pass_at_k(n, c, 8))
        p16s.append(_pass_at_k(n, c, min(16, n)))

    results["pass@1"]  = float(np.mean(p1s))
    results["pass@4"]  = float(np.mean(p4s))
    results["pass@8"]  = float(np.mean(p8s))
    results["pass@16"] = float(np.mean(p16s))
    logger.info(
        f"[eval] pass@1={results['pass@1']:.2%}  pass@4={results['pass@4']:.2%}  "
        f"pass@8={results['pass@8']:.2%}  pass@16={results['pass@16']:.2%}  "
        f"(over {len(eval_probs)} problems × {n_samples} samples)"
    )
    return results


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, tokenizer, path: Path, step: int, metrics: dict):
    ckpt_dir = path / f"step_{step:05d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    with open(ckpt_dir / "metrics.json", "w") as f:
        json.dump({"step": step, **metrics}, f, indent=2)
    logger.info(f"[ckpt] Saved checkpoint → {ckpt_dir}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

@dataclass
class DiversityGRPOConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    # Data
    dataset: str = "gsm8k"
    dataset_path: Optional[str] = None
    benchmark: str = "gsm8k"
    # Rollouts (from DARLING)
    n_rollouts: int = 8          # N methods per problem = N rollouts per group
    temperature: float = 0.8
    max_new_tokens: int = 8192
    # Reward
    lambda_div: float = 0.5      # weight on diversity reward
    embed_model: str = "openrouter"  # "openrouter" | "local"
    use_xml_steps: bool = False  # False = fast sentence splitting
    # GRPO hyperparams (from DARLING paper)
    lr: float = 1e-6
    warmup_ratio: float = 0.1
    clip_eps: float = 0.2
    kl_beta: float = 0.001
    # Training schedule
    n_problems_per_step: int = 32    # B = 32 problems × 8 rollouts = 256 global batch
    mini_batch: int = 8              # completions per backward pass (memory control)
    ref_batch_size: int = 2          # batched ref log-prob (keep small when co-locating vLLM)
    total_steps: int = 1000
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    # LoRA (LoRA keeps memory manageable)
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    # Checkpointing & logging
    output_dir: str = "results/rl_runs"
    save_every: int = 100
    log_every: int = 10
    # Resume: path to a saved step_XXXXX/ folder (LoRA adapter + tokenizer)
    resume_from: Optional[str] = None
    # Hardware
    gpu_id: int = 0
    seed: int = 42


def load_problems(dataset: str, dataset_path: Optional[str] = None) -> list[dict]:
    """
    Load problems with 'problem' and 'answer' fields.

    Supported sources:
      - Local .jsonl  (fields: problem/question + answer)
      - Local .parquet (DARLING format: problem + answer)
      - HuggingFace datasets: gsm8k, math, deepscaler, lighteval/MATH
      - Fallback: existing EDF cache
    """
    # ── 1. Explicit path ─────────────────────────────────────────────────────
    if dataset_path:
        path = Path(dataset_path)
        return _load_from_file(path)

    # ── 2. Local known paths ─────────────────────────────────────────────────
    data_dir = Path(__file__).parent.parent / "data" / "benchmarks"
    local_path = data_dir / f"{dataset}.jsonl"
    if local_path.exists():
        return _load_from_file(local_path)

    # ── 3. HuggingFace datasets ──────────────────────────────────────────────
    try:
        return _load_from_hf(dataset)
    except Exception as e:
        logger.warning(f"[data] HuggingFace load failed ({e}), trying EDF cache fallback")

    # ── 4. Fallback: existing R1 EDF cache ───────────────────────────────────
    cache = Path(__file__).parent.parent / "results" / "pas_cache" / "edf" / "edf_edf_nall_k3_full.jsonl"
    if cache.exists():
        logger.warning(f"[data] Using EDF cache: {cache}")
        return _load_from_file(cache)

    raise FileNotFoundError(f"No data found for dataset={dataset!r}")


def _load_from_file(path: Path) -> list[dict]:
    problems = []
    if path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(str(path))
        for _, row in df.iterrows():
            prob = row.get("problem", row.get("question", ""))
            ans = row.get("answer", row.get("solution", ""))
            if prob and ans:
                problems.append({"id": str(len(problems)), "problem": str(prob), "answer": str(ans)})
    else:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                prob = row.get("problem", row.get("question", ""))
                ans = row.get("answer", "")
                if prob and ans:
                    problems.append({"id": row.get("id", str(len(problems))),
                                     "problem": prob, "answer": ans})
    logger.info(f"[data] Loaded {len(problems)} problems from {path}")
    return problems


def _load_from_hf(dataset: str) -> list[dict]:
    """Load from HuggingFace datasets hub."""
    from datasets import load_dataset

    # Dataset name → (hf_name, split, problem_col, answer_col)
    HF_MAP = {
        "gsm8k":       ("openai/gsm8k", "train", "main", "question", "answer"),
        "math":        ("lighteval/MATH", "train", None, "problem", "solution"),
        "math500":     ("lighteval/MATH", "test", None, "problem", "solution"),
        "deepscaler":  ("agentica-org/DeepScaleR-Preview-Dataset", "train", None, "problem", "answer"),
        "aime":        ("AI-MO/aimo-validation-aime", "train", None, "problem", "answer"),
        "competition": ("deepmind/mathematics_dataset", "train", None, "question", "answer"),
    }

    if dataset not in HF_MAP:
        raise ValueError(f"Unknown HF dataset: {dataset!r}. Known: {list(HF_MAP.keys())}")

    entry = HF_MAP[dataset]
    hf_name, split, subset, prob_col, ans_col = entry

    kwargs = {"split": split}
    if subset:
        kwargs["name"] = subset

    ds = load_dataset(hf_name, **kwargs)
    problems = []
    for i, row in enumerate(ds):
        prob = row.get(prob_col, "")
        ans = row.get(ans_col, "")
        if prob and ans:
            # For GSM8K, clean the "#### 42" answer format
            if "####" in str(ans):
                ans = ans.split("####")[-1].strip()
            problems.append({"id": str(i), "problem": str(prob), "answer": str(ans)})
    logger.info(f"[data] Loaded {len(problems)} problems from HuggingFace {hf_name}:{split}")
    return problems


def train(cfg: DiversityGRPOConfig):
    from peft import LoraConfig, get_peft_model, TaskType

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # GPU selection: if CUDA_VISIBLE_DEVICES is already set externally (e.g. via
    # the launch script), respect it — that process already constrains which physical
    # GPU is visible, so cuda:0 is always the right logical index.
    # Only set CUDA_VISIBLE_DEVICES ourselves if it was NOT provided externally.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    device = "cuda:0"

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "training_log.jsonl"

    # ── Load tokenizer + model ───────────────────────────────────────────────
    logger.info(f"[init] Loading {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    start_step = 0
    if cfg.resume_from:
        from peft import PeftModel

        resume_path = Path(cfg.resume_from).resolve()
        metrics_file = resume_path / "metrics.json"
        if not resume_path.is_dir() or not metrics_file.exists():
            raise FileNotFoundError(f"resume_from must be a checkpoint dir with metrics.json: {resume_path}")
        with open(metrics_file) as f:
            start_step = int(json.load(f)["step"])
        logger.info(f"[init] Resuming LoRA from {resume_path} (start_step={start_step})")
        model = PeftModel.from_pretrained(model, str(resume_path), is_trainable=True)
        model.print_trainable_parameters()
    elif cfg.use_lora:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    model.to(device)
    model.gradient_checkpointing_enable()
    # Required for LoRA + gradient checkpointing: ensures the embedding output
    # has requires_grad=True so that checkpointed segments receive a grad-capable
    # input.  Without this, gradient checkpointing silently produces None grads.
    model.enable_input_require_grads()

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=0.0,
    )
    warmup_steps = int(cfg.total_steps * cfg.warmup_ratio)
    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=cfg.total_steps - warmup_steps, eta_min=0.0)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                              milestones=[warmup_steps])
    # Align LR schedule to resume step (optimizer state is fresh; LR curve matches step)
    for _ in range(start_step):
        scheduler.step()

    # ── Data ─────────────────────────────────────────────────────────────────
    problems = load_problems(cfg.dataset, cfg.dataset_path)

    # ── Training loop ────────────────────────────────────────────────────────
    from rl.diversity_reward import compute_group_rewards

    global_step = start_step
    if start_step >= cfg.total_steps:
        logger.warning(f"[train] start_step {start_step} >= total_steps {cfg.total_steps}; nothing to do.")
        return model, tokenizer
    optimizer.zero_grad()

    hot_lora_dir = output_dir / ".vllm_lora_gen"
    hot_lora_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.use_lora:
        logger.warning(
            "[train] use_lora=False: vLLM samples from the frozen HF base on disk; "
            "policy gradient targets a fine-tuned full model -- sampling is off-policy. "
            "Prefer --use_lora (default) for correct GRPO."
        )

    # vLLM is created at the start of each step and destroyed immediately after
    # rollouts.  Co-locating vLLM + HF + ref logits + backward on one 48GB card
    # OOMs; this costs one vLLM init per step (~10–15s) but is reliable.
    persistent_llm = None
    lora_slot_counter = start_step  # monotonically incrementing LoRA slot ID

    while global_step < cfg.total_steps:
        t0 = time.time()

        if persistent_llm is None:
            persistent_llm = create_vllm_engine(
                model_path=cfg.model_name,
                lora_rank=cfg.lora_r,
                use_lora=cfg.use_lora,
                max_new_tokens=cfg.max_new_tokens,
            )

        # Sample batch of problems (without replacement within epoch)
        batch = random.sample(problems, min(cfg.n_problems_per_step, len(problems)))

        # ── 1. Build all rollout prompts ─────────────────────────────────────
        flat_prompts: list[str] = []
        flat_meta: list[dict] = []   # {problem_idx, method_id, gold_answer}

        for pidx, prob in enumerate(batch):
            for method_id in range(1, cfg.n_rollouts + 1):
                flat_prompts.append(
                    build_chat_prompt(tokenizer, prob["problem"], method_id, cfg.n_rollouts)
                )
                flat_meta.append({
                    "problem_idx": pidx,
                    "method_id": method_id,
                    "gold_answer": prob["answer"],
                    "problem_id": prob["id"],
                })

        # ---- 2. Generate rollouts with persistent vLLM (on-policy via LoRA) -----
        # Save current LoRA weights to disk so vLLM loads the up-to-date policy.
        # Increment lora_slot_counter each step to force vLLM to reload the adapter
        # rather than serving stale cached weights from the previous step.
        lora_for_vllm: Optional[str] = None
        if cfg.use_lora:
            model.save_pretrained(str(hot_lora_dir))
            lora_for_vllm = str(hot_lora_dir)
        lora_slot_counter += 1
        completions = generate_rollouts(
            flat_prompts,
            model_path=cfg.model_name,
            lora_path=lora_for_vllm,
            lora_rank=cfg.lora_r,
            temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens,
            gpu_id=0,
            training_model=None,
            llm=persistent_llm,
            lora_slot_id=lora_slot_counter,
        )
        destroy_vllm_engine(persistent_llm)
        persistent_llm = None

        # ---- 3. Compute reference log-probs (BATCHED -- ~50x faster than sequential)
        # ref_batch_size=32 => 8 batched forward passes instead of 256 sequential.
        # No grad required so we can batch aggressively without OOM risk.
        model.eval()
        ref_log_probs = compute_ref_log_probs_batched(
            model, tokenizer, flat_prompts, completions,
            device=device, ref_batch_size=cfg.ref_batch_size,
        )

        # ── 4. Compute rewards per group ─────────────────────────────────────
        # Explicit no_grad: reward computation is purely scalar (numpy / API),
        # but this guard ensures no gradient graph is ever accidentally built
        # here — even if model-based step summarisation is added in future.
        flat_rewards: list[float] = []
        flat_correct: list[bool] = []
        group_div_scores: list[list[float]] = []

        with torch.no_grad():
            for pidx, prob in enumerate(batch):
                # All rollouts for this problem
                start = pidx * cfg.n_rollouts
                group_comps = completions[start: start + cfg.n_rollouts]
                rewards_g, correct_g, divs_g = compute_group_rewards(
                    completions=group_comps,
                    gold_answer=prob["answer"],
                    benchmark=cfg.benchmark,
                    lambda_div=cfg.lambda_div,
                    embed_model=cfg.embed_model,
                    use_xml_steps=cfg.use_xml_steps,
                )
                flat_rewards.extend(rewards_g)
                flat_correct.extend(correct_g)
                group_div_scores.append(divs_g)

        # ── 5. GRPO advantages ────────────────────────────────────────────────
        advantages = compute_grpo_advantages(flat_rewards, cfg.n_rollouts)

        # ── 6. Policy gradient update ────────────────────────────────────────
        # grpo_loss_and_backward processes mini-batches and calls backward()
        # immediately for each, keeping only one mini-batch's activations in
        # memory at a time instead of accumulating all 256 computation graphs.
        model.train()
        loss_val = grpo_loss_and_backward(
            model=model,
            tokenizer=tokenizer,
            prompts=flat_prompts,
            completions=completions,
            advantages=advantages,
            ref_log_probs=ref_log_probs,
            clip_eps=cfg.clip_eps,
            kl_beta=cfg.kl_beta,
            device=device,
            grad_accum=cfg.grad_accum,
            mini_batch=cfg.mini_batch,
        )

        if (global_step + 1) % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1
        elapsed = time.time() - t0
        n_correct = sum(flat_correct)
        n_total = len(flat_correct)
        pass_at_1 = np.mean(
            [any(flat_correct[i * cfg.n_rollouts: (i + 1) * cfg.n_rollouts])
             for i in range(len(batch))]
        )
        mean_div = np.mean([d for ds in group_div_scores for d in ds if d > 0] or [0.0])

        metrics = {
            "step": global_step,
            "loss": float(loss_val),
            "n_correct": n_correct,
            "n_total": n_total,
            "pass_at_1_approx": float(pass_at_1),
            "mean_diversity": float(mean_div),
            "lr": float(scheduler.get_last_lr()[0]),
            "elapsed_s": round(elapsed, 1),
        }

        if global_step % cfg.log_every == 0:
            logger.info(
                f"[step {global_step:4d}/{cfg.total_steps}] "
                f"loss={metrics['loss']:.4f}  "
                f"correct={n_correct}/{n_total}  "
                f"pass@1≈{pass_at_1:.2%}  "
                f"div={mean_div:.4f}  "
                f"lr={metrics['lr']:.2e}  "
                f"({elapsed:.1f}s)"
            )
            with open(log_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")

        if global_step % cfg.save_every == 0:
            # Save model weights
            save_checkpoint(model, tokenizer, output_dir, global_step, metrics)

            # COLD eval: pass@16 via approaches #1->8 x 2 (controllable diversity)
            logger.info(f"[eval] Running COLD pass@16 eval at step {global_step}...")
            tmp_ckpt = output_dir / f"step_{global_step:05d}"
            lora_slot_counter += 1
            try:
                eval_metrics = eval_pass_at_k(
                    cfg=cfg,
                    tokenizer=tokenizer,
                    lora_adapter_dir=str(tmp_ckpt),
                    problems=problems,
                    n_eval=min(50, len(problems)),
                    approach_cycles=2,
                    gpu_id=0,
                    training_model=model,
                    llm=None,
                    lora_slot_id=lora_slot_counter,
                )
                metrics.update(eval_metrics)
                # Overwrite metrics.json with eval scores included
                with open(tmp_ckpt / "metrics.json", "w") as f:
                    json.dump({"step": global_step, **metrics}, f, indent=2)
                logger.info(
                    f"[eval] pass@1={eval_metrics['pass@1']:.2%}  "
                    f"pass@8={eval_metrics['pass@8']:.2%}  "
                    f"pass@16={eval_metrics['pass@16']:.2%}"
                )
            except Exception as e:
                logger.warning(f"[eval] pass@k eval failed (non-fatal): {e}")

    # Final checkpoint + eval
    save_checkpoint(model, tokenizer, output_dir, global_step, metrics)
    destroy_vllm_engine(persistent_llm)
    persistent_llm = None
    torch.cuda.empty_cache()
    logger.info("[train] Done.")
    return model, tokenizer
