"""
COLD-RL Diversity Reward

Spec (Section 1.3–1.4):

  For attempt index i ∈ {1,...,I=16}, given a set of correct attempts
  C = {i : v(x,y_i)=1}:

    d_i = 1 − max_{j ∈ C, j < i} cos_sim_norm(e_i, e_j)

      where e_i  = embedding of the reasoning trace BEFORE the final answer
                   (last-layer pooling, no grad, frozen)
            norm  = cosine sims are min-max normalized within the rollout group
            d_i   = 0  if i==1, i∉C, |C|≤1, or no prior correct attempt exists

    w_i = λ_eff · (i−1) / (I−1)          [linearly growing with attempt index]

    r_i = v(x,y_i) · (1 + w_i · d_i)    [correctness-gated]
"""

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from generation.rollout import extract_answer


# ---------------------------------------------------------------------------
# Correctness grading
# ---------------------------------------------------------------------------

def is_correct(response: str, gold_answer: str, benchmark: str = "math") -> bool:
    pred = extract_answer(response)
    gold_raw = str(gold_answer).strip()
    if benchmark == "aime24":
        try:
            return int(pred.strip()) == int(gold_raw)
        except (ValueError, AttributeError):
            return pred.strip() == gold_raw
    gold_norm = extract_answer(gold_raw) if "\\boxed" in gold_raw else gold_raw
    return pred.strip() == gold_norm.strip()


# ---------------------------------------------------------------------------
# Reasoning trace extraction
# Spec: "embed strictly the reasoning trace (before the final answer line
#        delimiter, e.g., \n\n)"
# ---------------------------------------------------------------------------

def _extract_reasoning_trace(response: str) -> str:
    """Drop the final answer paragraph/line, keep only the reasoning."""
    # Try paragraph split: last paragraph is typically the boxed answer
    parts = response.rsplit("\n\n", 1)
    if len(parts) > 1 and "\\boxed" in parts[-1]:
        return parts[0].strip()
    # Fallback: drop last line containing \boxed
    lines = response.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        if "\\boxed" in lines[i] or "final answer" in lines[i].lower():
            return "\n".join(lines[:i]).strip()
    return response.strip()


# ---------------------------------------------------------------------------
# Embedding backend
# ---------------------------------------------------------------------------

_LOCAL_EMBED_MODEL = None
_QWEN3_EMBED_BUNDLE = None        # (tokenizer, model) for Qwen3-Embedding 8B
_QWEN3_SMALL_EMBED_BUNDLE = None  # (tokenizer, model) for Qwen3-Embedding-0.6B


def _get_local_model():
    global _LOCAL_EMBED_MODEL
    if _LOCAL_EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _LOCAL_EMBED_MODEL = SentenceTransformer(
            "intfloat/e5-small-v2", device="cpu"
        )
    return _LOCAL_EMBED_MODEL


def _embed_local(traces: list[str]) -> np.ndarray:
    mdl = _get_local_model()
    embs = mdl.encode(
        traces, normalize_embeddings=True,
        batch_size=64, show_progress_bar=False,
    )
    return embs.astype(np.float32)


def _embed_qwen3(traces: list[str]) -> np.ndarray:
    """Spec requirement: Qwen3-Embedding, last-layer pooling, no gradients."""
    global _QWEN3_EMBED_BUNDLE
    import torch
    from transformers import AutoTokenizer, AutoModel

    if _QWEN3_EMBED_BUNDLE is None:
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Embedding", trust_remote_code=True
        )
        mdl = AutoModel.from_pretrained(
            "Qwen/Qwen3-Embedding",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        mdl.eval()
        _QWEN3_EMBED_BUNDLE = (tok, mdl)

    tok, mdl = _QWEN3_EMBED_BUNDLE
    device = next(mdl.parameters()).device
    all_embs: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(traces), 16):
            batch = traces[start : start + 16]
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            hidden = mdl(**enc).last_hidden_state[:, 0, :]  # [CLS] pooling
            hidden = hidden.float().cpu().numpy()
            norms = np.linalg.norm(hidden, axis=1, keepdims=True) + 1e-8
            all_embs.append(hidden / norms)

    return np.concatenate(all_embs, axis=0).astype(np.float32)


def _embed_qwen3_small(traces: list[str]) -> np.ndarray:
    """Qwen3-Embedding-0.6B — same API as _embed_qwen3 but cheaper."""
    global _QWEN3_SMALL_EMBED_BUNDLE
    import torch
    from transformers import AutoTokenizer, AutoModel

    if _QWEN3_SMALL_EMBED_BUNDLE is None:
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True
        )
        mdl = AutoModel.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        mdl.eval()
        _QWEN3_SMALL_EMBED_BUNDLE = (tok, mdl)

    tok, mdl = _QWEN3_SMALL_EMBED_BUNDLE
    device = next(mdl.parameters()).device
    all_embs: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(traces), 16):
            batch = traces[start : start + 16]
            enc = tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            hidden = mdl(**enc).last_hidden_state[:, 0, :]  # [CLS] pooling
            hidden = hidden.float().cpu().numpy()
            norms = np.linalg.norm(hidden, axis=1, keepdims=True) + 1e-8
            all_embs.append(hidden / norms)

    return np.concatenate(all_embs, axis=0).astype(np.float32)


def _embed_openrouter(traces: list[str]) -> np.ndarray:
    import os, time
    import openai

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not set; use --embed_model local or qwen3"
        )
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=api_key
    )
    all_embs: list[np.ndarray] = []
    for start in range(0, len(traces), 256):
        batch = traces[start : start + 256]
        for attempt in range(3):
            try:
                resp = client.embeddings.create(
                    input=batch, model="Qwen/Qwen3-Embedding-8B"
                )
                embs = np.array([r.embedding for r in resp.data], dtype=np.float32)
                norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
                all_embs.append(embs / norms)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
    return np.concatenate(all_embs, axis=0)


def embed_traces(traces: list[str], embed_model: str = "local") -> np.ndarray:
    """
    Embed reasoning traces. Returns (N, d) L2-normalized float32 array.

    embed_model options:
      "local"       — intfloat/e5-small-v2 on CPU    (fast, no extra GPU)
      "qwen3-small" — Qwen/Qwen3-Embedding-0.6B GPU  (good quality, cheap)
      "qwen3"       — Qwen/Qwen3-Embedding 8B GPU    (spec requirement)
      "openrouter"  — Qwen3-Embedding-8B via API     (requires OPENROUTER_API_KEY)
    """
    if not traces:
        return np.zeros((0, 384), dtype=np.float32)
    if embed_model == "local":
        return _embed_local(traces)
    if embed_model == "qwen3-small":
        return _embed_qwen3_small(traces)
    if embed_model == "qwen3":
        return _embed_qwen3(traces)
    if embed_model == "openrouter":
        return _embed_openrouter(traces)
    raise ValueError(f"Unknown embed_model: {embed_model!r}")


# ---------------------------------------------------------------------------
# Core COLD-RL reward
# ---------------------------------------------------------------------------

def compute_cold_rewards(
    completions: list[str],
    gold_answer: str,
    benchmark: str = "math",
    lambda_eff: float = 0.5,
    embed_model: str = "local",
    I: int = 16,
) -> tuple[list[float], list[bool], list[float]]:
    """
    Compute r_i for each attempt i ∈ {1,...,I}.

    Returns
    -------
    rewards  : r_i = v_i · (1 + w_i · d_i)
    correct  : bool per attempt
    d_scores : raw d_i values (0 for non-contributing attempts)
    """
    assert len(completions) == I, (
        f"Expected exactly {I} completions, got {len(completions)}"
    )

    # 1. Grade correctness; build 1-indexed correct set C
    correct = [is_correct(c, gold_answer, benchmark) for c in completions]
    C = {i + 1 for i, ok in enumerate(correct) if ok}

    # 2. Per-attempt weights w_i = λ · (i-1)/(I-1)
    weights = [
        lambda_eff * (i - 1) / (I - 1) if I > 1 else 0.0
        for i in range(1, I + 1)
    ]

    # 3. d_i — only meaningful when ≥2 correct attempts exist and λ>0
    d_scores = [0.0] * I

    if len(C) >= 2 and lambda_eff > 0.0:
        # Extract reasoning traces (no final-answer paragraph)
        traces = [_extract_reasoning_trace(c) for c in completions]
        embs = embed_traces(traces, embed_model)  # (I, dim)

        # Cosine similarity matrix (embeddings are L2-normalized, so dot = cos_sim)
        sim_matrix = (embs @ embs.T).astype(np.float64)  # (I, I)

        # Min-max normalize within the rollout group using off-diagonal elements
        mask = ~np.eye(I, dtype=bool)
        off_diag = sim_matrix[mask]
        s_min, s_max = float(off_diag.min()), float(off_diag.max())
        if s_max > s_min:
            sim_norm = np.clip(
                (sim_matrix - s_min) / (s_max - s_min), 0.0, 1.0
            )
        else:
            sim_norm = np.clip(sim_matrix, 0.0, 1.0)

        for i in range(1, I + 1):          # 1-indexed attempt number
            idx = i - 1
            # Zero out per spec
            if i == 1 or (i not in C) or len(C) == 1:
                continue
            prior_correct = [j for j in C if j < i]
            if not prior_correct:
                continue
            sims = [sim_norm[idx, j - 1] for j in prior_correct]
            d_scores[idx] = float(1.0 - max(sims))

    # 4. r_i = v_i · (1 + w_i · d_i)
    rewards = [
        float(correct[i]) * (1.0 + weights[i] * d_scores[i])
        for i in range(I)
    ]

    return rewards, correct, d_scores


# ---------------------------------------------------------------------------
# Kept for backward compat with any code that still imports compute_group_rewards
# ---------------------------------------------------------------------------

def compute_group_rewards(
    completions: list[str],
    gold_answer: str,
    benchmark: str = "math",
    lambda_div: float = 0.5,
    embed_model: str = "local",
    use_xml_steps: bool = False,  # ignored — spec uses trace-level, not step-level
) -> tuple[list[float], list[bool], list[float]]:
    """Thin shim so old call-sites don't break during transition."""
    I = len(completions)
    return compute_cold_rewards(
        completions=completions,
        gold_answer=gold_answer,
        benchmark=benchmark,
        lambda_eff=lambda_div,
        embed_model=embed_model,
        I=I,
    )
