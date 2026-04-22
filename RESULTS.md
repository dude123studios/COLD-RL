# HugSim Experiment Results
Last updated: 2026-04-01 00:15

All pass@1 scores are on MATH500 (n=16 rollouts, temperature=0.6) unless noted.
Baseline target to beat: **0.5853**

---

## Completed Experiments

| Experiment ID | Group | Strategy | Problems | math500 | vs Baseline | gpqa | lcb | Status |
|---------------|-------|----------|----------|---------|-------------|------|-----|--------|
| `baseline_3b` | A | None | 0 | **0.5853** | — | — | — | Done |
| `sft_1s_3b` | A | SFT textbook, old HP | 12.5k | 0.3210 | -0.264 | — | — | Done |
| `sft_1s_fixed_2k_3b` | J | SFT textbook, correct HP | 2k | 0.3250 | -0.260 | — | — | Done |
| `sdft_2k_3b` | I | Self-distillation (SDFT) | 2k | 0.5770 | -0.008 | — | — | Done |
| `sft_teacher_2k_3b` | J | Rejection-sampling SFT | 2k | **0.5954** | **+0.010** | — | — | Done |
| `pas_me_k1_2k_3b` | K | PAS-ME k=1 (INVALID) | 2k | 0.3869 | -0.199 | — | — | Done — textbook contamination |
| `pas_me_k3_2k_3b` | K | PAS-ME k=3 (INVALID) | 2k | 0.4445 | -0.141 | — | — | Done — textbook contamination |
| `pas_me_k1_2k_3b_v2` | L | PAS-ME k=1, two-phase | 2k | 0.5640 | -0.021 | 0.055 | 0.478 | Done |
| `pas_me_k2_2k_3b_v2` | L | PAS-ME k=2, two-phase | 2k | 0.5426 | -0.043 | 0.062 | 0.480 | Done |
| `pas_me_k3_2k_3b_v2` | L | PAS-ME k=3, two-phase | 2k | 0.5408 | -0.044 | 0.059 | 0.472 | Done |
| `pas_me_k5_2k_3b_v2` | L | PAS-ME k=5, two-phase | 2k | TBD | — | TBD | TBD | Dataset building |
| `sft_teacher_2k_7b` | N | Rejection-sampling SFT (7B) | 2k | **0.6431** | **+0.058** | 0.201 | 0.520 | Done |
| `pas_me_k3_2k_7b` | N | PAS-ME k=3 (7B) | 2k | 0.5811 | -0.004 | 0.096 | 0.497 | Done |

### Key findings
1. Textbook LaTeX always causes regression regardless of method.
2. Model-generated (rejection-sampled) SFT: `sft_teacher_2k_3b` beats baseline (+1.0%).
3. **PAS-ME v2 disappointing**: k=1=0.564, k=2=0.543, k=3=0.541, k=5=TBD — all below baseline and teacher SFT.
   Clear monotonic degradation with more alternatives:
   - 3B teacher not strong enough for high-quality branch-point alternatives
   - Branch-point alternatives may introduce reasoning inconsistencies
   - GPQA trend: k=1=0.055, k=2=0.062, k=3=0.059 (no clear trend, all near 0.06)
4. 7B scales well: `sft_teacher_2k_7b` = 0.643 (+5.8% over baseline). `pas_me_k3_2k_7b` = 0.581 (-0.4% vs baseline) — PAS degradation persists at 7B scale.

---

## In Progress

| Experiment ID | Group | Stage | GPU | Notes |
|---------------|-------|-------|-----|-------|
| `pas_me_k5_2k_3b_v2` | L | Training | 2 | 27% (113/420 steps), ~38 min |
| `grpo_2k_3b` | O | Training | 1 | ~13h left |
| `baseline_7b_v2` | N | Eval math500 | 0 | Just launched, ~1h |
| `sft_teacher_5k_3b` | M | Eval math500 | 3 | 75% (5968/8000), ~10 min |
| `sft_teacher_10k_3b` | M | Dataset gen | 4 | 37% of 66968 prompts, ~2.5h |

### Why v2 vs original K group
Original Group K trained with textbook originals mixed in. Group L v2 uses the
fixed two-phase strategy (teacher_gen → PAS augmentation from model-gen solutions).

---

## Pending (not yet started)

### Group M — Data efficiency sweep
| Experiment ID | Problems | Strategy | Status |
|---------------|----------|----------|--------|
| `sft_teacher_500_3b` | 500 | teacher_gen | Queued on GPU 2 |
| `sft_teacher_1k_3b` | 1000 | teacher_gen | Queued on GPU 2 |
| `sft_teacher_5k_3b` | 5000 | teacher_gen | Dataset gen running (GPU 3, ~3.5h) |
| `sft_teacher_10k_3b` | 10000 | teacher_gen | Dataset gen running (GPU 4, ~3.5h) |
| `pas_me_k3_500_3b` | 500 | PAS-ME k=3 | Queued on GPU 2 |
| `pas_me_k3_1k_3b` | 1000 | PAS-ME k=3 | Queued on GPU 2 |
| `pas_me_k3_5k_3b` | 5000 | PAS-ME k=3 | Watcher script → GPU 3 after 5k teacher |
| `pas_me_k3_10k_3b` | 10000 | PAS-ME k=3 | Watcher script → GPU 4 after 10k teacher |

### Group O — GRPO
| Experiment ID | Notes | Status |
|---------------|-------|--------|
| `grpo_2k_3b` | GRPO from scratch | Training (GPU 1, ~16h left) |
| `grpo_from_pas_2k_3b` | PAS-ME k=3 → GRPO | Pending (adapter preserved) |

### Group N — 7B scaling
| Experiment ID | Notes | Status |
|---------------|-------|--------|
| `baseline_7b_v2` | 7B baseline | Auto-run after pas_me_k3_2k_7b |
| `sft_teacher_2k_7b` | Rejection-sampling SFT on 7B | **Done** (0.643) |
| `pas_me_k3_2k_7b` | PAS-ME k=3 on 7B | Training (GPU 0, ~45 min) |

---

## Eval Benchmarks

All future experiments eval on three benchmarks:

| Benchmark | Problems | Type | Scoring |
|-----------|---------|------|---------|
| `math500` | 500 | Competition math | `\boxed{}` extraction |
| `gpqa_diamond` | 198 | Graduate science (Physics 86, Chem 93, Bio 19) | Letter A/B/C/D extraction |
| `livecodebench` | 175 | Competitive coding (easy 43 / medium 52 / hard 80) | Code execution vs public test cases |

GPQA data files available:
- `data/benchmarks/gpqa_diamond.jsonl` — 198 problems (test/eval)
- `data/benchmarks/gpqa_main.jsonl` — 448 problems (superset of diamond)
- `data/benchmarks/gpqa_extended.jsonl` — 546 problems (largest, use for training augmentation)
- `data/benchmarks/livecodebench.jsonl` — 175 problems

Source: OpenAI simple-evals (diamond), Wanfq/gpqa mirror (main/extended)

---

## Infrastructure Issues Encountered

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `sft_1s_3b` regressed to 0.321 | Textbook LaTeX distribution gap | Use rejection-sampled model-gen solutions |
| SDFT didn't improve (0.577) | Teacher = student, no signal gain | Needs strictly stronger teacher |
| `pas_me_k1` regressed to 0.387 | `include_original=True` pulled textbook originals | Two-phase fix: teacher_gen → PAS from those |
| `pas_me_k3` regressed to 0.445 | Same contamination | Two-phase fix applied in v2 |
| PAS alternatives non-diverse | "Make MINIMAL changes" prompt fought diversity | Removed constraint; use branch-point prompt |
| k=1/k=2 OOM during training | `device_map=auto` picked GPU 3 (occupied by another user) | Restart with `CUDA_VISIBLE_DEVICES=1` |
| Teacher_gen re-run for each k | Different cache key per k, same underlying data | Pre-seeded k3/k5 caches from k1 |

---

## GPU State (as of 2026-03-31)

| GPU | Status | Owner |
|-----|--------|-------|
| 0 | Available (vLLM generation uses it temporarily) | hugsim |
| 1 | **Free — assigned to Group L training** | hugsim |
| 2 | Free | available |
| 3 | ~Full (internvl training) | yuxin |
| 4 | ~Full (internvl training) | yuxin |
| 5 | ~Full (MemAgent vLLM) | other |
| 6 | ~Full (MemAgent vLLM) | other |
| 7 | ~Full (MemAgent vLLM) | other |

---

## Paper Story (emerging)

1. **Distribution matters more than method**: Textbook SFT regresses; model-gen SFT works
2. **PAS (v2) hypothesis**: Branch-point augmentation from model-gen solutions should
   improve over plain rejection-sampling SFT, especially on harder problems
3. **Data efficiency**: PAS should beat SFT at smaller n_problems (main paper claim)
4. **Generalization**: GPQA + LiveCodeBench evals check that math SFT doesn't hurt
   general reasoning or coding ability

Next critical results needed: Group L k=1,2,3,5 — these establish whether PAS v2 works.
