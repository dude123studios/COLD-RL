"""
COLD-RL Phase 2 — Water-filling allocator (spec §2.2)

Given a calibrated Δ(n, i) table, greedily assigns a total budget of k
draws across attempt indices 1..I to maximise joint pass@k:

    pass@k = 1 − Π_i (1 − P̂(n_i*, i))

Algorithm (greedy water-filling):
    n_i = 0  for all i
    for t = 1..k:
        i* = argmax_i  Δ(n_i, i)      # marginal gain from one more draw at index i
        n_{i*} += 1

Reports pass@k for k ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}.

Also implements three comparison allocators (spec §E4):
  - uniform:          n_i = k / I  (floor, remainder distributed to first indices)
  - bernoulli_wf:     marginal p_i · (1−p_i)^{n_i}  (closed-form approx)
  - oracle:           uses true per-problem p_i(x) from the R-draw pool

Usage
-----
    python -m evaluation.water_filling \
        --calib results/calibration/e1_7b.json \
        --k-values 1 2 4 8 16 32 64 128 \
        --out results/water_filling/e1_7b_alloc.json
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

K_REPORT = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


# ---------------------------------------------------------------------------
# Core allocator
# ---------------------------------------------------------------------------

def greedy_water_fill(
    delta: np.ndarray,   # (n_max, I) marginal gains; delta[n, i] = P̂(n+1,i)−P̂(n,i)
    phat:  np.ndarray,   # (n_max+1, I) cumulative estimates
    k: int,
) -> tuple[np.ndarray, float]:
    """
    Greedy allocation of k draws across I indices.

    Returns
    -------
    n_alloc   : (I,) int allocation vector
    joint_pass: scalar joint pass@k estimate
    """
    n_max, I = delta.shape
    n_alloc = np.zeros(I, dtype=np.int64)

    for _ in range(k):
        best_i, best_gain = -1, -1.0
        for i in range(I):
            ni = int(n_alloc[i])
            if ni < n_max:
                gain = float(delta[ni, i])
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
        if best_i < 0:
            break
        n_alloc[best_i] += 1

    # joint pass@k = 1 − Π_i (1 − P̂(n_i*, i))
    log_fail = 0.0
    for i in range(I):
        p = float(phat[int(n_alloc[i]), i])
        p = max(0.0, min(1.0, p))
        log_fail += math.log(max(1e-300, 1.0 - p))
    joint_pass = float(1.0 - math.exp(log_fail))

    return n_alloc, joint_pass


def uniform_allocate(
    phat: np.ndarray,
    k: int,
) -> tuple[np.ndarray, float]:
    """n_i = floor(k/I); distribute remainder to first (k % I) indices."""
    I = phat.shape[1]
    n_alloc = np.full(I, k // I, dtype=np.int64)
    for i in range(k % I):
        n_alloc[i] += 1

    n_max = phat.shape[0] - 1
    log_fail = 0.0
    for i in range(I):
        p = float(phat[min(int(n_alloc[i]), n_max), i])
        p = max(0.0, min(1.0, p))
        log_fail += math.log(max(1e-300, 1.0 - p))
    return n_alloc, float(1.0 - math.exp(log_fail))


def bernoulli_water_fill(
    phat: np.ndarray,
    k: int,
) -> tuple[np.ndarray, float]:
    """
    Greedy using marginal gain  p_i · (1−p_i)^{n_i}  where p_i = P̂(1, i).
    Closed-form approximation for geometric decay.
    """
    I = phat.shape[1]
    n_alloc = np.zeros(I, dtype=np.int64)
    p1 = phat[1, :].copy()  # P̂(1, i) — single-draw success probability

    for _ in range(k):
        gains = np.array([
            p1[i] * ((1.0 - p1[i]) ** n_alloc[i]) for i in range(I)
        ])
        best_i = int(np.argmax(gains))
        n_alloc[best_i] += 1

    n_max = phat.shape[0] - 1
    log_fail = 0.0
    for i in range(I):
        p = float(phat[min(int(n_alloc[i]), n_max), i])
        p = max(0.0, min(1.0, p))
        log_fail += math.log(max(1e-300, 1.0 - p))
    return n_alloc, float(1.0 - math.exp(log_fail))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def run_allocator_sweep(
    calib_path: str,
    k_values: Optional[list[int]] = None,
    out_path:  Optional[str] = None,
) -> dict:
    """
    Load a calibration JSON and run all four allocators for each k.

    Returns a dict:
      results[k] = {
        "empirical_wf":  {"allocation": [...], "pass_at_k": ...},
        "uniform":       {...},
        "bernoulli_wf":  {...},
      }
    """
    if k_values is None:
        k_values = K_REPORT

    calib = json.loads(Path(calib_path).read_text())
    phat  = np.array(calib["phat"],  dtype=np.float64)   # (n_max+1, I)
    delta = np.array(calib["delta"], dtype=np.float64)   # (n_max, I)
    meta  = calib["meta"]

    results = {"meta": meta, "k_values": k_values, "allocations": {}}

    header = f"{'k':>5}  {'empirical_wf':>14}  {'uniform':>10}  {'bernoulli_wf':>14}"
    print(header)
    print("-" * len(header))

    for k in k_values:
        alloc_emp,  pass_emp  = greedy_water_fill(delta, phat, k)
        alloc_uni,  pass_uni  = uniform_allocate(phat, k)
        alloc_bern, pass_bern = bernoulli_water_fill(phat, k)

        results["allocations"][str(k)] = {
            "empirical_wf":  {"allocation": alloc_emp.tolist(),  "pass_at_k": pass_emp},
            "uniform":       {"allocation": alloc_uni.tolist(),  "pass_at_k": pass_uni},
            "bernoulli_wf":  {"allocation": alloc_bern.tolist(), "pass_at_k": pass_bern},
        }
        print(f"{k:>5}  {pass_emp:>14.4f}  {pass_uni:>10.4f}  {pass_bern:>14.4f}")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(results, indent=2))
        logger.info(f"[water_filling] Results → {out_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="COLD-RL water-filling allocator")
    ap.add_argument("--calib", required=True,
                    help="Path to calibration JSON from evaluation/calibration.py")
    ap.add_argument("--k-values", nargs="+", type=int, default=K_REPORT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_allocator_sweep(
        calib_path=args.calib,
        k_values=args.k_values,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
