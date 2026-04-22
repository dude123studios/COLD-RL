# HugSim Paper Experiment Plan: Path-Augmented SFT (PAS)

Last updated: 2026-03-31

---

## Core Finding (established)

Standard SFT with textbook solutions always regresses (0.585 → 0.321) regardless of
hyperparameters — it's a **distribution gap**, not a tuning problem. Textbook LaTeX solutions
have high perplexity under instruct models; training on them causes catastrophic forgetting.

**All competitive training methods must use model-generated solutions.**

Proof:
- `sft_1s_3b` (12.5k, textbook, lr=2e-5): 0.321
- `sft_1s_fixed_2k_3b` (2k, textbook, lr=1e-4): 0.325 — correct HP doesn't fix the gap
- `sft_teacher_2k_3b` (2k, model-generated, lr=1e-4): **0.595** — first improvement

---

## Paper Thesis

PAS minimal-edit (rejection-sampling base + local path augmentation) achieves the best
pass@1 per unique problem seen, beating standard rejection-sampling SFT and GRPO at equal
data budgets, without requiring a stronger external teacher model.

---

## Current Results Scoreboard

| Experiment ID | Method | Problems | pass@1 | Notes |
|---------------|--------|----------|--------|-------|
| `baseline_3b` | No training | 0 | **0.585** | Target to beat |
| `sft_1s_3b` | SFT textbook, old HP | 12.5k | 0.321 | Regression |
| `sft_1s_fixed_2k_3b` | SFT textbook, correct HP | 2k | 0.325 | Regression — proves distribution gap |
| `sdft_2k_3b` | SDFT (self-distillation) | 2k | 0.577 | Near-baseline — same-model teacher is weak |
| `sft_teacher_2k_3b` | Rejection-sampling SFT | 2k | **0.595** | First improvement |
| `pas_me_k1_2k_3b` | PAS-ME k=1 (INVALID) | 2k | 0.387 | Regression — textbook contamination |
| `pas_me_k3_2k_3b` | PAS-ME k=3 (INVALID) | 2k | ~0.38 | Regression expected — same contamination |

"INVALID" = `include_original=True` pulled in textbook solutions as the "original" before
the two-phase fix was applied. Results are kept as ablation evidence showing why two-phase matters.

---

## Experiment Groups

### Group A–H — Legacy baselines (wrong hyperparams + textbook solutions)

These ran with lr=2e-5, epochs=1, max_length=4096, textbook ground-truth solutions.
**None are expected to improve over baseline.** Kept in registry for ablation story.

Key IDs: `sft_1s_3b`, `sft_3s_3b`, `sft_3diverse_3b`, `grpo_3b`, `pas_k3_3b_full`,
k-sweep (B), size-sweep (C), context ablation (D), 7B (E), GRPO-from-PAS (F), rare problem (H).

---

### Group I — SDFT (Self-Distillation Fine-Tuning) on 2k

Teacher = same 3B model conditioned on the correct solution. Student = same model trained
via forward KL against saved teacher logits (top-50). Result: 0.577 — near baseline.

**Conclusion**: Same-model teacher doesn't provide a strong enough signal. SDFT only helps
when teacher >> student. Useful negative result for the paper.

| ID | Status | Result |
|----|--------|--------|
| `sdft_2k_3b` | Done | 0.577 |
| `sdft_pas_2k_3b` | Stalled (OpenRouter failures) | — |

---

### Group J — Corrected SFT ablation (establishes the key causal claim)

Isolates the distribution gap vs hyperparameter effect.

| ID | Strategy | HP | Result | Takeaway |
|----|----------|----|--------|----------|
| `sft_1s_fixed_2k_3b` | Textbook solutions | Correct (lr=1e-4, ep=2) | 0.325 | HP doesn't fix the gap |
| `sft_teacher_2k_3b` | Teacher-gen rejection sampling | Correct | 0.595 | Distribution is the cause |

---

### Group K — INVALIDATED: PAS minimal-edit with textbook contamination

Built before the two-phase fix. `include_original=True` included textbook solutions.
Results: k=1 → 0.387, k=3 → ~0.38. **Superseded by Group L.**

Used in paper as: "naively mixing model-gen PAS alternatives with textbook originals
still causes regression — the original solution distribution matters, not just the alternatives."

---

### Group L — PAS minimal-edit FIXED (two-phase) ← **run next**

**Strategy (fixed `pas_minimal_edit`):**
1. Phase 1 (teacher_gen): model generates n=8 solutions per problem via sampling,
   keep first correct one (rejection sampling). All training solutions are now model-generated.
2. Phase 2 (minimal_edit): same 3B model locally via vLLM minimally rewrites those
   model-gen solutions to use different mathematical pathways (k alternatives per problem).
3. Result: every training example is in the model's own distribution + k diverse paths.

**K sweep on 2k problems:**

| ID | k | Training examples (est.) | Status |
|----|---|--------------------------|--------|
| `pas_me_k1_2k_3b_v2` | 1 | ~2k | Pending |
| `pas_me_k2_2k_3b_v2` | 2 | ~3k | Pending |
| `pas_me_k3_2k_3b_v2` | 3 | ~4k | Pending |
| `pas_me_k5_2k_3b_v2` | 5 | ~6k | Pending |

**Hyperparams (all Group L):** lr=1e-4, epochs=2, max_length=8192, warmup_ratio=0.03,
lora_r=16, lora_alpha=16, batch_size=1, grad_accum=16.

```bash
python -m orchestration.run_experiment --groups L
```

---

### Group M — Data efficiency sweep

How many unique problems does each method need to beat baseline (0.585)?

Methods: `teacher_gen` (1 solution/problem) vs `pas_minimal_edit k=3` (4 solutions/problem).

| ID | Method | Problems | Status |
|----|--------|----------|--------|
| `sft_teacher_500_3b` | teacher_gen | 500 | Pending |
| `sft_teacher_1k_3b` | teacher_gen | 1000 | Pending |
| `sft_teacher_2k_3b` | teacher_gen | 2000 | **Done (0.595)** |
| `sft_teacher_5k_3b` | teacher_gen | 5000 | Pending |
| `sft_teacher_10k_3b` | teacher_gen | 10000 | Pending |
| `pas_me_k3_500_3b` | PAS-ME k=3 | 500 | Pending |
| `pas_me_k3_1k_3b` | PAS-ME k=3 | 1000 | Pending |
| `pas_me_k3_2k_3b_v2` | PAS-ME k=3 | 2000 | Pending (Group L) |
| `pas_me_k3_5k_3b` | PAS-ME k=3 | 5000 | Pending |
| `pas_me_k3_10k_3b` | PAS-ME k=3 | 10000 | Pending |

Key figure: pass@1 vs log(n_problems) curve showing crossover point.

```bash
python -m orchestration.run_experiment --groups M
```

---

### Group N — 7B model sweep

Verify methods generalize to larger model. 7B baseline will be higher so absolute gains
may differ; relative ordering should hold.

| ID | Method | Status |
|----|--------|--------|
| `baseline_7b_v2` | No training | Pending |
| `sft_teacher_2k_7b` | teacher_gen | Pending |
| `pas_me_k3_2k_7b` | PAS-ME k=3 | Pending |

Note: 7B needs ~16GB+ VRAM for LoRA training. Verify hardware before scheduling.

```bash
python -m orchestration.run_experiment --groups N
```

---

### Group O — GRPO baseline and GRPO-from-PAS pipeline

Tests whether PAS warm-start improves GRPO convergence or final performance.

| ID | Method | Depends on | Status |
|----|--------|-----------|--------|
| `grpo_2k_3b` | GRPO from scratch (2k) | — | Pending |
| `grpo_from_pas_2k_3b` | PAS-ME k=3 → GRPO | `pas_me_k3_2k_3b_v2` | Pending |

```bash
python -m orchestration.run_experiment --groups O
```

---

## Paper Tables

### Table 1 — Main comparison (2k problems, 3B model)

| Method | Unique problems | Training rows | pass@1 |
|--------|----------------|--------------|--------|
| Baseline (no training) | 0 | 0 | 0.585 |
| SFT textbook (old HP) | 12.5k | 12.5k | 0.321 |
| SFT textbook (correct HP) | 2k | 2k | 0.325 |
| SDFT (self-distillation) | 2k | 2k | 0.577 |
| SFT teacher-gen | 2k | 2k | **0.595** |
| PAS-ME k=1 (two-phase) | 2k | ~2k | *TBD* |
| PAS-ME k=2 (two-phase) | 2k | ~3k | *TBD* |
| PAS-ME k=3 (two-phase) | 2k | ~4k | *TBD* |
| PAS-ME k=5 (two-phase) | 2k | ~6k | *TBD* |
| GRPO (2k) | 2k | 2k | *TBD* |
| PAS-ME k=3 → GRPO | 2k | ~4k+RL | *TBD* |

### Table 2 — Data efficiency (pass@1 vs problems)

| Problems | SFT teacher-gen | PAS-ME k=3 |
|----------|----------------|-----------|
| 500 | *TBD* | *TBD* |
| 1000 | *TBD* | *TBD* |
| 2000 | 0.595 | *TBD* |
| 5000 | *TBD* | *TBD* |
| 10000 | *TBD* | *TBD* |

### Table 3 — Scaling to 7B

| Method | 3B pass@1 | 7B pass@1 |
|--------|----------|----------|
| Baseline | 0.585 | *TBD* |
| SFT teacher-gen (2k) | 0.595 | *TBD* |
| PAS-ME k=3 (2k) | *TBD* | *TBD* |

### Table 4 — Ablations (all 2k, 3B)

| Ablation axis | Condition | pass@1 |
|---------------|-----------|--------|
| Distribution | Textbook solutions | 0.325 |
| Distribution | Model-generated (teacher_gen) | 0.595 |
| PAS original quality | Textbook + ME alts (INVALID) | 0.387 |
| PAS original quality | Teacher-gen + ME alts (two-phase) | *TBD* |
| Teacher strength | 3B self-play (SDFT) | 0.577 |
| Teacher strength | 3B self-play (PAS-ME) | *TBD* |
| Context mode | Full solution (PAS-ME k=3) | *TBD* |
| Context mode | First sentence only | *(Group D, needs rerun w/ correct HP)* |

---

## Hyperparameter Standard (all Group L+)

```yaml
lr: 1.0e-4
epochs: 2
max_length: 8192
warmup_ratio: 0.03
lora_r: 16
lora_alpha: 16
lora_dropout: 0.05
batch_size: 1
grad_accum: 16  # effective batch = 16
```

Groups A–I used lr=2e-5, epochs=1, max_length=4096 — treat as historical baselines only.

---

## Execution Order (critical path)

```
1. [ ] Wait for pas_me_k3_2k_3b (Group K) to finish → record regression ~0.38
2. [ ] Run Group L (k sweep, fixed two-phase)       ← UNBLOCKED
3. [ ] Run Group O p1 (grpo_2k_3b) in parallel with L
4. [ ] Run Group M p1 (500, 1k) once Phase 1 teacher_gen cache exists
5. [ ] Run Group O p2 (grpo_from_pas) once L is done
6. [ ] Run Group M p2-3 (5k, 10k) — long tail, GPU-intensive
7. [ ] Run Group N (7B) — verify VRAM first
```

## What's Blocked / At Risk

- **Group O (GRPO)**: requires `training/grpo_train.py` to support `init_from_lora`.
  Verify before scheduling `grpo_from_pas_2k_3b`.
- **Group H (rare problem)**: requires `generation/rare_problem.py` — not yet written.
  Lower priority; can defer to revision.
- **Group N (7B)**: verify available VRAM. LoRA training at max_length=8192 needs ~24GB.
- **Group D (context ablation)**: existing experiments used wrong HP. Need v2 variants
  with lr=1e-4 if context mode ablation goes in final paper.
- **`sdft_pas_2k_3b`**: stalled due to OpenRouter failures. Can skip — SDFT story is
  already told by `sdft_2k_3b`.
