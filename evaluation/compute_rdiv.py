"""
R-Div and S-Div metric computation.

R-Div: proportion of rollouts for a problem that are semantically novel
       relative to all previously seen rollouts (ordered by generation).

S-Div: same as R-Div but restricted to correct rollouts only.

Step extraction IS used here (at eval time) because model rollouts are
messy free-form text. For PAS data generation, step extraction is skipped
because clean solutions are used directly.
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np

EMBED_CACHE_DIR = Path(__file__).parent.parent / "results" / "embeddings_cache"
STEP_EXTRACT_CACHE_DIR = Path(__file__).parent.parent / "results" / "step_cache"


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def embed_text(text: str, embedder=None) -> np.ndarray:
    """Embed a single text with disk caching keyed by content hash."""
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _text_hash(text)
    cache_path = EMBED_CACHE_DIR / f"{key}.npy"

    if cache_path.exists():
        return np.load(str(cache_path))

    if embedder is None:
        embedder = _get_embedder()

    emb = embedder.encode([text], normalize_embeddings=True)[0]
    np.save(str(cache_path), emb)
    return emb


def embed_texts_batch(texts: list[str], embedder=None, batch_size: int = 256) -> np.ndarray:
    """
    Embed a list of texts with disk caching. Batches uncached texts together
    for a large speedup over calling embed_text one-by-one.
    Returns embeddings in the same order as input texts.
    """
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    keys = [_text_hash(t) for t in texts]
    cache_paths = [EMBED_CACHE_DIR / f"{k}.npy" for k in keys]

    # Identify which texts need encoding
    missing_idx = [i for i, p in enumerate(cache_paths) if not p.exists()]

    if missing_idx:
        if embedder is None:
            embedder = _get_embedder()
        missing_texts = [texts[i] for i in missing_idx]

        # Encode in batches
        all_embs = []
        for start in range(0, len(missing_texts), batch_size):
            batch = missing_texts[start: start + batch_size]
            all_embs.append(embedder.encode(batch, normalize_embeddings=True))
        new_embs = np.concatenate(all_embs, axis=0)

        for idx, emb in zip(missing_idx, new_embs):
            np.save(str(cache_paths[idx]), emb)

    # Load all (now guaranteed cached)
    return np.array([np.load(str(p)) for p in cache_paths])


def extract_steps_for_eval(text: str) -> list[str]:
    """
    Decompose a model rollout into logical steps for R-Div computation.

    Heuristic approach (no LLM call needed for step boundaries):
    - Split on numbered step patterns ("Step 1:", "1.", "First,", etc.)
    - Split on paragraph breaks with substantial content
    - Filter out very short segments (<30 chars)
    """
    # Try numbered steps first
    step_pattern = re.compile(
        r"(?:^|\n)(?:Step\s+\d+[:.)]|\d+[.)])\s+", re.IGNORECASE
    )
    parts = step_pattern.split(text)

    if len(parts) > 2:
        steps = [p.strip() for p in parts if len(p.strip()) > 30]
        return steps if steps else [text]

    # Fallback: split on paragraph breaks
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
    if paragraphs:
        return paragraphs

    return [text]


def compute_rdiv(
    rollouts: list[str],
    gold_answer: Optional[str] = None,
    threshold: float = 0.25,
    embedder=None,
) -> dict:
    """
    Compute R-Div for a single problem's rollouts.

    A rollout is "novel" if its maximum cosine similarity to all
    previously-seen rollout embeddings is below (1 - threshold).
    I.e., threshold=0.25 means a rollout must differ by at least
    0.25 in cosine distance from everything seen before.

    Args:
        rollouts: list of n rollout strings for one problem
        gold_answer: if provided, compute S-Div over correct rollouts only
        threshold: novelty threshold (higher = stricter)
        embedder: optional pre-loaded SentenceTransformer

    Returns:
        dict with rdiv, sdiv, novelty_scores, correct_mask
    """
    from generation.rollout import extract_answer

    if embedder is None:
        embedder = _get_embedder()

    embeddings = embed_texts_batch(rollouts, embedder)

    novelty_scores = []
    novel_mask = []
    seen_embeddings = []

    for i, emb in enumerate(embeddings):
        if not seen_embeddings:
            novelty_scores.append(1.0)
            novel_mask.append(True)
        else:
            seen = np.array(seen_embeddings)
            similarities = np.dot(seen, emb)
            max_sim = float(np.max(similarities))
            novelty = 1.0 - max_sim
            novelty_scores.append(novelty)
            novel_mask.append(novelty >= threshold)
        seen_embeddings.append(emb)

    rdiv = float(np.mean(novel_mask))

    # S-Div: restricted to correct rollouts
    sdiv = None
    correct_mask = None
    if gold_answer is not None:
        correct_mask = [extract_answer(r) == gold_answer for r in rollouts]
        correct_and_novel = [c and n for c, n in zip(correct_mask, novel_mask)]
        n_correct = sum(correct_mask)
        sdiv = float(sum(correct_and_novel) / n_correct) if n_correct > 0 else 0.0

    return {
        "rdiv": rdiv,
        "sdiv": sdiv,
        "novelty_scores": novelty_scores,
        "novel_mask": novel_mask,
        "correct_mask": correct_mask,
    }


def compute_rdiv_dataset(
    rollouts_by_problem: dict[str, list[str]],
    problems: Optional[list[dict]] = None,
    threshold: float = 0.25,
) -> dict:
    """
    Compute R-Div and S-Div aggregated over all problems.

    Args:
        rollouts_by_problem: problem_id -> list of rollouts
        problems: list of {id, answer} for S-Div; if None, S-Div is skipped
        threshold: novelty threshold

    Returns:
        dict with rdiv_mean, sdiv_mean, rdiv_per_problem, sdiv_per_problem
    """
    embedder = _get_embedder()
    answer_map = {p["id"]: p["answer"] for p in problems} if problems else {}

    # Pre-embed all rollouts in one batched call (much faster than one-by-one)
    all_texts = [r for rollouts in rollouts_by_problem.values() for r in rollouts]
    n_total = len(all_texts)
    print(f"[rdiv] Embedding {n_total} rollouts across {len(rollouts_by_problem)} problems...", flush=True)
    embed_texts_batch(all_texts, embedder)  # warms cache; compute_rdiv reads from cache
    print(f"[rdiv] Embeddings done, computing novelty scores...", flush=True)

    rdiv_per_problem = {}
    sdiv_per_problem = {}

    for i, (pid, rollouts) in enumerate(rollouts_by_problem.items()):
        gold = answer_map.get(pid)
        result = compute_rdiv(rollouts, gold_answer=gold, threshold=threshold, embedder=embedder)
        rdiv_per_problem[pid] = result["rdiv"]
        if result["sdiv"] is not None:
            sdiv_per_problem[pid] = result["sdiv"]
        if (i + 1) % 100 == 0:
            print(f"[rdiv] {i+1}/{len(rollouts_by_problem)} problems scored", flush=True)

    return {
        "rdiv_mean": float(np.mean(list(rdiv_per_problem.values()))),
        "sdiv_mean": float(np.mean(list(sdiv_per_problem.values()))) if sdiv_per_problem else None,
        "rdiv_per_problem": rdiv_per_problem,
        "sdiv_per_problem": sdiv_per_problem if sdiv_per_problem else None,
        "threshold": threshold,
        "n_problems": len(rdiv_per_problem),
    }
