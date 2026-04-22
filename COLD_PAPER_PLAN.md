# COLD Paper — Full Experimental Plan

> Last updated: 2026-04-18  
> Status: living document — tick off runs as they complete

---

## Key invariant
**If your GRPO-4B replication matches DARLING Table 9 within ±0.5pp on AIME25 + HMMT25 + OlympiadBench + Brumo → cite their numbers directly.**  
Math suite follows **DARLING’s reported benchmarks** (including **Brumo**); we are **not** running a separate AMC track.

---

## §5.2 — Controllability verification

> Two rows, **same checkpoint**, different eval prompts.  
> The p-value comparison between C-CTRL-1 and GRPO is the **central empirical claim** of the paper.

| Run | Model | Benchmarks | Key config | Matches | Est. GPU-h | Type |
|-----|-------|-----------|-----------|---------|-----------|------|
| C-CTRL-1 | Qwen3-4B-Base | AIME25, HMMT25, OlympiadBench, Brumo | train T=1.0, **eval T=0.6** top-p=0.95, K=8, λ=0.5, **prefix ABSENT** at eval | Should match GRPO numbers | ~14h | **critical / new** |
| C-CTRL-2 | Qwen3-4B-Base | Same | same ckpt as C-CTRL-1, **eval T=0.6** top-p=0.95, K=8, **prefix PRESENT** at eval | — | eval only | **critical / new** |

---

## §5.3 — Main math results (Qwen3-4B-Base & 8B-Base)

| Run | Model | Benchmarks | Key config | Matches published | Est. GPU-h | Type |
|-----|-------|-----------|-----------|------------------|-----------|------|
| BASE-4B | Qwen3-4B-Base | AIME25, HMMT25, OlympiadBench, Brumo | No training — **eval T=0.6**, top-p=0.95, n=256 | DARLING Table 9 exact | eval only | reuse published |
| GRPO-4B | Qwen3-4B-Base | Same | DeepScaleR 10k, n=8, **train T=1.0 / eval T=0.6** top-p=0.95, lr=1e-6, β=0, 10 epochs, clip=(0.2,0.2), token-level avg, **NO std-norm** | DARLING Table 9 — must match exactly | ~16h | **reuse published / critical** |
| BASE-8B | Qwen3-8B-Base | Same | No training — **eval T=0.6**, top-p=0.95, n=256 | DARLING Table 9/10 | eval only | reuse published |
| GRPO-8B | Qwen3-8B-Base | Same | Same as GRPO-4B (**train T=1.0 / eval T=0.6**) | DARLING | ~28h | **reuse published / critical** |
| DARLING-4B | Qwen3-4B-Base | Same | Published checkpoint / reported numbers | DARLING Table 9 — cite directly | 0 | reuse published |
| **COLD-4B ⭐** | Qwen3-4B-Base | AIME25, HMMT25, OlympiadBench, Brumo | DeepScaleR 10k, K=8, **train T=1.0 / eval T=0.6** top-p=0.95, lr=1e-6, β=0, **λ ramp 0→0.5 epochs 1-3**, e5-small-v2, token-level avg, NO std-norm, clip=(0.2,0.2), 10 epochs | — | ~18h | **new / critical** |
| **COLD-8B ⭐** | Qwen3-8B-Base | Same | Same as COLD-4B (**train T=1.0 / eval T=0.6**) | — | ~32h | **new / critical** |
| TEMP-4B | Qwen3-4B-Base | Same | GRPO-4B checkpoint, **eval T=1.4** (intentional high-temp experiment), K=8 | Derived from GRPO-4B — no extra training | eval only | reuse published |

> **Brumo note:** DARLING reports **Brumo** in the same math table as AIME / HMMT / OlympiadBench (Table 9). Use **the same eval protocol** (T=0.6, top-p=0.95, n=256, math-verify) for all four.  
> **Citing:** If GRPO-4B matches DARLING within 0.5pp on **AIME25, HMMT25, OlympiadBench, and Brumo** (Table 9), cite Li et al. for baselines on all four; otherwise extend training / diagnosis until alignment holds.

---

## §5.4 — Scientific reasoning (SciKnowEval, Qwen3-8B-Base)

> SDPO uses Qwen3-8B, reports at **1h and 5h wall-clock**. Match this exactly — report at both checkpoints.

| Run | Model | Tasks | Key config | Matches published | Est. GPU-h | Type |
|-----|-------|-------|-----------|------------------|-----------|------|
| GRPO-SCI | Qwen3-8B-Base | SciKnowEval Chem, Physics | **train T=1.0**, avg@16 eval, **5h wall-clock** | SDPO Table 1 Qwen3-8B+GRPO row | ~5h | reuse published |
| SDPO-SCI | Qwen3-8B-Base | Same | Cite reported numbers only | SDPO Table 1 | 0 | reuse published |
| **COLD-SCI ⭐** | Qwen3-8B-Base | SciKnowEval Chem, Physics | K=8, **train T=1.0**, avg@16 eval, λ ramp, e5-small-v2, **5h wall-clock** | — | ~5h | **new / critical** |

---

## §5.5 — Instruction following (Llama-3.1-8B-Instruct)

| Run | Model | Benchmarks | Key config | Matches published | Est. GPU-h | Type |
|-----|-------|-----------|-----------|------------------|-----------|------|
| BASE-LLM | Llama-3.1-8B-Instruct | AlpacaEval 2.0, ArenaHard v2.0, EQ-Bench, NoveltyBench | No training | DARLING Table 1 instruct row | 0 | reuse published |
| GRPO-LLM | Llama-3.1-8B-Instruct | Same | WildChat 10k, n=8, **train T=1.0 / eval T=0.6** top-p=0.9, lr=1e-6, β=0.001, Athene-RM-8B reward, 10 epochs, 1024 tok max | DARLING Table 1 GRPO row | ~8h | reuse published |
| DIVPO-LLM | Llama-3.1-8B-Instruct | Same | Cite directly | DARLING Table 1 | 0 | reuse published |
| DARLING-LLM | Llama-3.1-8B-Instruct | Same | Cite directly | DARLING Table 1 | 0 | reuse published |
| GRPO-UNL | Llama-3.1-8B-Instruct | Same | Cite directly | DARLING Table 1 | 0 | reuse published |
| **COLD-LLM ⭐** | Llama-3.1-8B-Instruct | Same | WildChat 10k, K=8, **train T=1.0 / eval T=0.6** top-p=0.9, lr=1e-6, β=0.001, Athene-RM-8B reward, **λ ramp**, e5-small-v2, 1024 tok max, 10 epochs | — | ~10h | **new / critical** |

---

## §6 — Ablations (all Qwen3-4B-Base, AIME25+HMMT25+OlympiadBench+Brumo)

| Ablation | Sweep values | Fixed config | Runs | Est. GPU-h | Type |
|----------|-------------|-------------|------|-----------|------|
| **A1 — Conditioning mechanism** | ① no prefix (GRPO) ② random 4-token id ③ "#k" only ④ strategy-specific prefixes ⑤ COLD full | λ=0.5, K=8, 10 epochs | 5 | ~70h | ablation |
| **A2 — Diversity metric** ★ | ① none ② 4-gram ratio ③ semantic classifier (DARLING) ④ embed full solution ⑤ embed trace only (COLD) | λ=0.5, K=8, full COLD prefix | 5 | ~70h | **ablation / critical** |
| **A3 — Reward fusion** | ① additive (r = v + λ·Div) ② multiplicative no gate ③ COLD gated multiplicative | λ=0.5, K=8, embed trace | 3 | ~42h | ablation |
| **A4 — Number of rollouts K** | K ∈ {2, 4, 6, 8, 12} | λ=0.5, full COLD prefix, embed trace | 5 | ~60h | ablation |
| **A5 — Diversity coefficient λ** | λ ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 1.0} | K=8, full COLD prefix, embed trace | 6 | ~84h | ablation |
| **A6 — Reward hacking analysis** | Shows 4-gram is gameable in math — run A2-4gram to completion, manual inspection 50 rollouts | Uses A2 ④ checkpoint | 0 extra | 0 extra | **ablation / critical** |
| **A7 — λ curriculum schedule** | ① fixed λ=0.5 ② step at epoch 3 ③ COLD linear ramp | K=8, full COLD prefix, embed trace | 3 | ~42h | ablation |
| **A8 — Reward normalization** | COLD + std-norm vs COLD − std-norm (default) | K=8, full COLD prefix, λ=0.5 | 2 | ~28h | **ablation — matches DARLING §6.3** |

---

## §7 — Analysis runs

| Run | Model | Purpose | Config | Est. GPU-h | Type |
|-----|-------|---------|--------|-----------|------|
| AN1 — Strategy probing | Qwen3-4B-Base (COLD ckpt) | Manual categorization of strategies by attempt index. Shows index k → strategy type learned. | Sample 100 OlympiadBench × 8 rollouts. Human or GPT-4o strategy labeling. | eval only | new |
| AN2 — Scale study | 4B + 8B | COLD improvement over GRPO vs model size | Use COLD-4B and COLD-8B checkpoints already trained | 0 extra | new |
| AN3 — COLD+Temp combination | Qwen3-4B-Base (COLD ckpt) | Verify COLD and temperature are orthogonal | **Intentional high-temp sweep:** eval COLD-4B at T ∈ {0.6, 0.8, 1.0, 1.2, 1.4}, K=8 | eval only | new |

> **No 14B runs anywhere.** Scale study (§7.2) uses 4B + 8B only — extracted from already-trained checkpoints. Zero extra compute.

---

## Complete hyperparameter reference

> Must be identical to DARLING/SDPO where baselines are shared.

> ⚠️ **Temperature rule:**  
> **Train T = 1.0 always** (needed for rollout diversity during RL — never change this).  
> **Eval T = 0.6, top-p=0.95 always** (DARLING Table 8 — matches all published baselines).  
> The only intentional exceptions are: **TEMP-4B** (eval T=1.4) and **AN3** (T sweep 0.6→1.4).  
> Any time you see "T=1.0" in a run's key config it refers to the **training rollout temperature**, not the eval setting.

| Param | Math (4B/8B) | Science (8B) | Instruction (8B-Instruct) | Source lock |
|-------|-------------|-------------|--------------------------|------------|
| Training data | DeepScaleR 10k (filtered) | SciKnowEval reasoning subset | WildChat 10k | DARLING §4.1 / SDPO §3 |
| Rollout n / K | 8 | 8 | 8 | match both papers |
| **Train temperature** | **T=1.0** | **T=1.0** | **T=1.0** | rollout generation only |
| **Eval temperature** | **T=0.6, top-p=0.95** | **avg@16** | **T=0.6, top-p=0.9** | DARLING Table 8 — default for all evals |
| Max response length | 8192 | 4096 | 1024 | DARLING Table 6/7 |
| Max prompt length | 1024 | 1024 | 512 | DARLING Table 6 |
| Learning rate | 1e-6 | 1e-6 | 1e-6 | match both papers |
| KL coefficient β | 0.0 | 0.0 | 0.001 | DARLING Table 6/7 |
| KL loss type | N/A | N/A | low_var_kl | DARLING Table 6 |
| Clip ratio ε | (0.2, 0.2) | (0.2, 0.2) | (0.2, 0.2) | DARLING Table 7 |
| Loss averaging | token-level | token-level | token-level | DARLING §3.2 |
| **Std-norm in advantage** | **OFF** | **OFF** | **OFF** | DARLING §6.3 |
| Epochs | 10 | until 5h wall-clock | 10 | DARLING Table 6 / SDPO §3 |
| Batch (prompts × rollouts) | 32 × 8 = 256 | 32 × 8 = 256 | 32 × 8 = 256 | match DARLING |
| Reward model | math-verify (binary) | binary verifier | Athene-RM-8B | DARLING §4.1 |
| Inference engine | vLLM | vLLM | vLLM | match both papers |
| Training framework | verl | verl | verl | match both papers |
| pass@k eval n | 256 per problem | avg@16 | NoveltyBench defaults | DARLING §5.1 |

### COLD-only hyperparameters

| Param | Value | Note |
|-------|-------|------|
| λ | 0.5 (ramped 0→0.5 over epochs 1-3) | all domains |
| Embed model | e5-small-v2 (frozen, 33M params) | all domains |
| Embed input | reasoning trace only (before `\boxed{}`) | math/science |
| Embed input (instruct) | full response | instruction |
| Diversity formula | `1 − min_{j≠k, j∈C} cos_sim(e_k, e_j)` | all domains |
| Prefix | `"[Attempt #k — use a new method]"` | all domains |

---

## Brumo benchmark — eval config (DARLING-reported)

| Param | Value | Note |
|-------|-------|------|
| Dataset | **Brumo** (exact split / preprocessing as in DARLING code or paper supplement) | Same contamination / formatting assumptions as DARLING's other math benchmarks — do **not** substitute AMC. |
| Eval temperature | T=0.6, top-p=0.95 | DARLING Table 8 |
| n per problem | 256 | Matches DARLING §5.1 |
| Verifier | math-verify (HuggingFace) | Same stack as AIME / HMMT / OlympiadBench in DARLING |
| Answer format | Follow Brumo task definition (open-ended numeric / expression as in benchmark) | Not multiple-choice |
| pass@k | k ∈ {1, 2, 4, 8, 16, 32, 64, 128} | Matches DARLING Fig 6 |

> ✅ **Why Brumo instead of AMC:** DARLING **published** Brumo scores in Table 9, so a matching GRPO replication unlocks **direct citation** on that column. **AMC is out of scope** for this paper version (no parallel AMC table / verifier work).

---

## Compute budget summary

| Category | Est. GPU-h | Notes |
|----------|-----------|-------|
| Critical new runs (§5.2–5.5) | ~115h | C-CTRL-1, COLD-4B, COLD-8B, COLD-SCI, COLD-LLM |
| Reuse published / eval only | ~0–57h | GRPO reruns only if replication fails |
| Ablations (§6) | ~396h total | Sequential on 2×H100 ≈ few days |
| Analysis (§7) | eval only | Uses already-trained checkpoints |
| **Total new training** | **~115h critical + ~396h ablations** | **~58h wall-clock on 2×H100 for critical path** |

---

## Match condition for citing DARLING directly

1. Run GRPO-4B with exact hyperparams above.
2. If **AIME25 / HMMT25 / OlympiadBench / Brumo** match DARLING Table 9 within **±0.5pp** → cite their numbers, state: *"our GRPO replication matches Li et al. (2025) within 0.3pp; we use their reported numbers."*
3. If match fails → must run DARLING yourselves (adds ~16h).
4. **Brumo** uses the same replication-and-cite rule as AIME25, HMMT25, and OlympiadBench (no separate AMC track).

---

## Match condition for citing SDPO directly

1. Run GRPO-SCI on SciKnowEval at 5h wall-clock.
2. If matches SDPO Table 1 GRPO row → cite SDPO directly for GRPO and SDPO numbers; only report COLD-SCI.

---

## Status tracker

| Run ID | Status | GPU-h used | Notes |
|--------|--------|-----------|-------|
| C-CTRL-1 | ⬜ pending | — | |
| C-CTRL-2 | ⬜ pending | — | |
| BASE-4B | ⬜ pending | — | |
| GRPO-4B | ⬜ pending | — | |
| BASE-8B | ⬜ pending | — | |
| GRPO-8B | ⬜ pending | — | |
| COLD-4B | ⬜ pending | — | |
| COLD-8B | ⬜ pending | — | |
| TEMP-4B | ⬜ pending | — | |
| GRPO-SCI | ⬜ pending | — | |
| COLD-SCI | ⬜ pending | — | |
| GRPO-LLM | ⬜ pending | — | |
| COLD-LLM | ⬜ pending | — | |
| A1 (×5) | ⬜ pending | — | |
| A2 (×5) | ⬜ pending | — | critical — A2-4gram also feeds A6 |
| A3 (×3) | ⬜ pending | — | |
| A4 (×5) | ⬜ pending | — | |
| A5 (×6) | ⬜ pending | — | |
| A6 | ⬜ pending | — | no extra training — uses A2 ④ ckpt |
| A7 (×3) | ⬜ pending | — | |
| A8 (×2) | ⬜ pending | — | |
| AN1 | ⬜ pending | — | eval only |
| AN2 | ⬜ pending | — | 0 extra — uses COLD-4B + COLD-8B ckpts |
| AN3 | ⬜ pending | — | eval only |
