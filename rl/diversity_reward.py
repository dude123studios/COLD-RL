"""
Diversity reward computation for Diversity-GRPO.

Pipeline per rollout group (N traces for one problem):
  1. Grade correctness (binary).
  2. For each correct trace: extract steps into a list of text chunks.
  3. Embed all steps with Qwen3-Embedding-8B via OpenRouter (or local fallback).
  4. For trace i: diversity_score_i = max over all steps s in trace i,
                                       max over all j≠i, all steps s' in trace j:
                                       cosine_distance(s, s')
     i.e. the single most "unique" step in trace i compared to all other traces.
  5. Combined reward: r_i = r_correct_i + lambda_div * (r_correct_i * diversity_score_i)
     (diversity is GATED by correctness)
"""

import os
import re
import time
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Answer extraction & correctness grading (reuse existing logic)
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from generation.rollout import extract_answer


def is_correct(response: str, gold_answer: str, benchmark: str = "gsm8k") -> bool:
    pred = extract_answer(response)
    gold_raw = str(gold_answer).strip()
    if benchmark == "aime24":
        try:
            return int(pred.strip()) == int(gold_raw)
        except (ValueError, AttributeError):
            return pred.strip() == gold_raw
    # Competition / HF math: gold may be plain or contain \\boxed{}
    if benchmark in ("math", "deepscaler", "math500", "gsm8k"):
        gold_norm = extract_answer(gold_raw) if "\\boxed" in gold_raw else gold_raw
        return pred.strip() == gold_norm.strip()
    return pred == gold_raw


# ---------------------------------------------------------------------------
# Step extraction
# ---------------------------------------------------------------------------

STEP_CACHE = Path(__file__).parent.parent / "results" / "rl_step_cache"


def _split_steps_simple(text: str) -> list[str]:
    """
    Fast sentence-level step extraction (no extra model call).
    Split on sentence boundaries, keeping only non-trivial chunks.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    steps = [s.strip() for s in sentences if len(s.strip()) > 20]
    return steps if steps else [text[:500]]


def _extract_steps_xml(response: str, model_name: str = None) -> list[str]:
    """
    Parse XML step tags if the model returned them.
    Falls back to simple splitting if not found.
    """
    matches = re.findall(r"<step[^>]*>(.*?)</step>", response, re.DOTALL)
    if matches:
        return [m.strip() for m in matches if m.strip()]
    return _split_steps_simple(response)


def extract_steps(trace: str, use_xml: bool = False) -> list[str]:
    """Extract steps from a trace. use_xml=True parses <step> tags."""
    if use_xml:
        return _extract_steps_xml(trace)
    return _split_steps_simple(trace)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

EMBED_CACHE = Path(__file__).parent.parent / "results" / "rl_embed_cache"
EMBED_CACHE.mkdir(parents=True, exist_ok=True)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def embed_batch_openrouter(texts: list[str], model: str = "Qwen/Qwen3-Embedding-8B") -> np.ndarray:
    """
    Embed a list of texts using OpenRouter's embedding endpoint.
    Returns shape (len(texts), embedding_dim), L2-normalized.
    Uses per-text disk cache to avoid re-embedding on reruns.
    """
    import openai

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set; use local embeddings instead (--embed_model local)")

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Per-text caching to avoid paying twice for the same step
    embeddings = []
    missing_idxs = []
    missing_texts = []

    for i, text in enumerate(texts):
        cache_path = EMBED_CACHE / f"{_hash(text)}.npy"
        if cache_path.exists():
            embeddings.append(np.load(str(cache_path)))
        else:
            embeddings.append(None)
            missing_idxs.append(i)
            missing_texts.append(text)

    if missing_texts:
        # Batch in chunks of 256
        CHUNK = 256
        fetched = []
        for start in range(0, len(missing_texts), CHUNK):
            chunk = missing_texts[start:start + CHUNK]
            for attempt in range(3):
                try:
                    resp = client.embeddings.create(input=chunk, model=model)
                    fetched.extend([r.embedding for r in resp.data])
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)

        for idx, emb_list in zip(missing_idxs, fetched):
            emb = np.array(emb_list, dtype=np.float32)
            # L2 normalise
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            cache_path = EMBED_CACHE / f"{_hash(texts[idx])}.npy"
            np.save(str(cache_path), emb)
            embeddings[idx] = emb

    return np.stack(embeddings)


_LOCAL_EMBED_MODEL = None


def _get_local_embed_model():
    """Lazy singleton: load SentenceTransformer once and reuse across all calls."""
    global _LOCAL_EMBED_MODEL
    if _LOCAL_EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _LOCAL_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _LOCAL_EMBED_MODEL


def embed_batch_local(texts: list[str]) -> np.ndarray:
    """Local embedding fallback using sentence-transformers (singleton model)."""
    model = _get_local_embed_model()
    embs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return embs.astype(np.float32)


def embed_steps(steps: list[str], embed_model: str = "openrouter") -> np.ndarray:
    """
    Embed a list of step strings.
    embed_model: "openrouter" (Qwen3-Embedding-8B) | "local" (MiniLM)
    """
    if not steps:
        return np.zeros((0, 384), dtype=np.float32)
    if embed_model == "local":
        return embed_batch_local(steps)
    # OpenRouter Qwen3-Embedding-8B
    return embed_batch_openrouter(steps, model="Qwen/Qwen3-Embedding-8B")


# ---------------------------------------------------------------------------
# Diversity score computation
# ---------------------------------------------------------------------------

def compute_diversity_scores(
    step_embeddings: list[Optional[np.ndarray]],
) -> list[float]:
    """
    For each trace i with step embeddings matrix E_i (shape [n_steps_i, d]):

      diversity_score_i = max over steps s in E_i of:
                            min over j≠i, all steps s' in E_j:
                              cosine_distance(s, s')

    This is the "most isolated step" formulation:
      - For each step s in trace i, compute its nearest-neighbor cosine distance
        to any step across all other traces (how isolated is this step?).
      - The diversity score is the maximum such nearest-neighbor distance —
        i.e., the single most unique step in trace i gets the score.

    Traces that introduce at least one step that is far from any step in all
    other rollouts receive a high diversity reward.

    Args:
        step_embeddings: list of N elements. Each is either:
            - None  (incorrect trace, no diversity reward)
            - np.ndarray of shape (n_steps, embed_dim), L2-normalised
    Returns:
        list[float] diversity scores, 0.0 for incorrect traces.
    """
    n = len(step_embeddings)
    scores = []

    for i in range(n):
        if step_embeddings[i] is None or len(step_embeddings[i]) == 0:
            scores.append(0.0)
            continue

        E_i = step_embeddings[i]  # (n_steps_i, d)

        # Collect all step embeddings from other traces
        other_embs = []
        for j in range(n):
            if j == i or step_embeddings[j] is None or len(step_embeddings[j]) == 0:
                continue
            other_embs.append(step_embeddings[j])

        if not other_embs:
            scores.append(0.0)
            continue

        E_others = np.concatenate(other_embs, axis=0)  # (total_other_steps, d)

        # For each step s in trace i: nearest-neighbor cosine distance across E_others
        # sim matrix: (n_steps_i, total_other_steps)
        sim_matrix = E_i @ E_others.T         # cosine similarities (normalised)
        dist_matrix = 1.0 - sim_matrix        # cosine distances ∈ [0, 2]

        # Nearest-neighbor distance for each step in trace i
        nn_dists = dist_matrix.min(axis=1)    # (n_steps_i,)

        # Most isolated step: the step with the highest nearest-neighbor distance
        diversity_score_i = float(nn_dists.max())
        scores.append(diversity_score_i)

    return scores


# ---------------------------------------------------------------------------
# Full group reward computation
# ---------------------------------------------------------------------------

_APPROACH_FRAMING_RE = re.compile(
    r"^[\s\S]*?(?:approach|method|solution|strategy|technique|algebraic|geometric"
    r"|arithmetic|number.theoretic|combinatorial|direct|indirect)[^\n]*[\n.!?]+",
    re.IGNORECASE,
)


def _strip_approach_framing(text: str) -> str:
    """
    Remove any opening sentence that merely re-states the approach label
    (e.g. "Using an algebraic approach, ..." or "Method 3: ...").
    This prevents the model from gaming diversity by varying only the framing
    words while producing identical mathematics.
    """
    # Only strip a single leading sentence (≤ 200 chars); if longer, leave untouched.
    m = _APPROACH_FRAMING_RE.match(text)
    if m and m.end() < 200:
        return text[m.end():].strip()
    return text


def compute_group_rewards(
    completions: list[str],
    gold_answer: str,
    benchmark: str = "gsm8k",
    lambda_div: float = 0.5,
    embed_model: str = "openrouter",
    use_xml_steps: bool = False,
) -> tuple[list[float], list[bool], list[float]]:
    """
    Compute combined rewards for all N completions of one problem.

    Reward hacking prevention:
      - Diversity is computed ONLY on the mathematical content of each step,
        after stripping any opening sentence that merely re-states the
        approach label (e.g. "Using an algebraic approach...").  This prevents
        the model from earning diversity reward by varying framing language
        while producing identical mathematics.
      - Diversity is gated on correctness: incorrect traces get 0 diversity reward.
      - All values are plain floats / numpy arrays — no PyTorch computation graph
        is involved here, so no gradients can leak back through the reward signal.

    Returns:
        rewards      : combined reward (correctness + gated diversity)
        correct_mask : bool per completion
        div_scores   : raw diversity score per completion (0 if incorrect)
    """
    correct_mask = [is_correct(c, gold_answer, benchmark) for c in completions]

    # Only compute diversity for correct traces
    all_steps: list[Optional[list[str]]] = []
    all_flat_steps: list[str] = []
    step_slices: list[Optional[tuple[int, int]]] = []

    offset = 0
    for i, (comp, ok) in enumerate(zip(completions, correct_mask)):
        if ok:
            # Strip framing before step extraction to prevent surface-level gaming
            content = _strip_approach_framing(comp)
            steps = extract_steps(content, use_xml=use_xml_steps)
            all_steps.append(steps)
            step_slices.append((offset, offset + len(steps)))
            all_flat_steps.extend(steps)
            offset += len(steps)
        else:
            all_steps.append(None)
            step_slices.append(None)

    # Embed all steps in one batch
    step_embs_by_trace: list[Optional[np.ndarray]] = []
    if all_flat_steps:
        flat_embs = embed_steps(all_flat_steps, embed_model=embed_model)
        for i, slc in enumerate(step_slices):
            if slc is not None:
                step_embs_by_trace.append(flat_embs[slc[0]:slc[1]])
            else:
                step_embs_by_trace.append(None)
    else:
        step_embs_by_trace = [None] * len(completions)

    div_scores = compute_diversity_scores(step_embs_by_trace)

    rewards = [
        float(ok) + lambda_div * float(ok) * d
        for ok, d in zip(correct_mask, div_scores)
    ]

    return rewards, correct_mask, div_scores
