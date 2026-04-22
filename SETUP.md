# COLD-RL — Fresh GPU Cluster Setup & Experiment Guide

> **What this is:** COLD (Controllable On-demand Learned Diversity) trains a Diversity-GRPO RL policy on top of Qwen models so the model can produce *K* **genuinely distinct** solution approaches on demand while staying correct.  
> At inference, the model uses standard prompts for best-answer, or `"Approach #k"` prompts for *K* diverse solutions — diversity is **controllable, not always-on**.

---

## 1. Hardware requirements

| Config | Min VRAM | Recommended |
|--------|----------|-------------|
| 7B training | 48 GB per GPU | 48–80 GB |
| 4B training | 24 GB per GPU | 40–80 GB |
| 8B training | 48 GB per GPU | 80 GB |
| Eval only | 24 GB | any A100/H100 |

Each run occupies **one GPU** (`CUDA_VISIBLE_DEVICES=X`). Four jobs can run in parallel on four GPUs.

---

## 2. What the Git repo does *not* include (by design)

The repository is **source + scripts only**. These are **gitignored** and appear on a fresh machine after setup or first run:

| Artifact | How it is obtained |
|----------|-------------------|
| **Base model weights** (Qwen, Llama, …) | Downloaded automatically by **Transformers** / **vLLM** from the HuggingFace Hub on first use (`~/.cache/huggingface` unless you set `HF_HOME`). |
| **DeepScaleR / MATH / GSM8K** | Loaded via **`datasets`** from the Hub in `load_problems()` — no manual drop-in required for standard names. Optional: `python scripts/download_benchmarks.py`. |
| **Training checkpoints & LoRA** | Written under `results/` during `rl.run_rl` — not in git. |
| **Logs, eval caches, vLLM temp LoRA dir** | Under `logs/`, `results/**/eval_cache/`, `.vllm_lora_gen/` — ignored. |
| **Python env** | Create with `python -m venv .venv` + `pip install -r requirements.txt`. |

So **clone + pip + first training step** pulls everything heavy; you do not need to copy tarballs or weights through Git.

---

## 3. One-time setup on a fresh node

```bash
# 1. Clone
git clone https://github.com/dude123studios/COLD-RL.git
cd COLD-RL

# 2. Python environment (Python 3.10–3.13 OK)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) math verifier — needed for DeepScaleR / MATH grading
pip install math-verify  # or: pip install antlr4-python3-runtime==4.11.0 latex2sympy2
```

> **Note:** `vllm` ≥ 0.6 is required. If you hit `EngineCoreClient` init errors, pin to a known-good vLLM version:
> ```bash
> pip install "vllm==0.8.5"
> ```

---

## 4. Environment variables

```bash
# Required for openrouter-based embeddings (more accurate diversity reward)
export OPENROUTER_API_KEY="sk-or-..."

# Reduces CUDA allocator fragmentation — ALWAYS set this
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Optional: set HF cache dir if disk space is limited elsewhere
export HF_HOME=/scratch/hf_cache
export TRANSFORMERS_CACHE=/scratch/hf_cache
```

---

## 5. How COLD works (theory in 90 seconds)

### Standard GRPO recap
At each training step:
1. Sample *B* problems × *K* rollouts → 256 completions
2. Compute correctness reward (binary pass/fail via math-verify)
3. Normalize rewards within each group of *K* rollouts (GRPO advantage)
4. Policy gradient update

### What COLD adds
1. **Method prefix in the prompt:** each rollout uses prompt `"Solve using mathematical approach #k of K (algebraic, geometric, number-theoretic…)"`. This makes the model learn to produce method-specific solutions.
2. **Diversity reward:** for each correct rollout *i*, compute the max cosine distance between any step in trace *i* and any step in any other trace in the same group.  
   ```
   diversity_score_i = max_{j≠i} max_{s in trace_i, s' in trace_j} cosine_dist(embed(s), embed(s'))
   ```
3. **Gated combined reward:**  
   ```
   r_i = r_correct_i + λ × r_correct_i × diversity_score_i
   ```
   Diversity reward is **zero** if the answer is wrong — the model can't earn diversity bonus by being creatively wrong.

### What λ controls
| λ | Effect |
|---|--------|
| 0.0 | Pure GRPO — no diversity signal (baseline) |
| 0.1 | Mild diversity nudge |
| 0.5 | Primary setting — balanced correctness + diversity |
| 0.7 | Strong diversity pressure |

### Why this beats DARLING
DARLING always trains for diversity — at inference the model is always diverse, which can hurt accuracy on standard prompts. COLD trains **on-demand diversity**: standard prompt → best answer; method prompt → diverse solutions. This gives the best of both worlds.

---

## 6. Full experiment list (parallel-safe)

### 6a. Critical new results (paper §5)

These are the runs that **prove the paper's claims**. Run all in parallel.

| ID | GPU | Command |
|----|-----|---------|
| **GRPO-baseline** | 0 | `bash scripts/run_grpo_baseline.sh` |
| **COLD λ=0.5** | 1 | `bash scripts/run_cold_lam05.sh` |
| **COLD λ=0.1** | 2 | `bash scripts/run_cold_lam01.sh` |
| **COLD λ=0.7** | 3 | `bash scripts/run_cold_lam07.sh` |

Or use the single launcher that does all four:

```bash
bash restart2.sh
```

### 6b. Model scale experiments (§5.3)

Run **after** 7B runs confirm methodology works; use Qwen3 models for paper tables.

| ID | Model | GPU | Notes |
|----|-------|-----|-------|
| COLD-4B | Qwen/Qwen3-4B-Base | 0 | ~18 GPU-h |
| GRPO-4B | Qwen/Qwen3-4B-Base | 1 | Replication of DARLING — cite directly if ±0.5pp |
| COLD-8B | Qwen/Qwen3-8B-Base | 2 | ~32 GPU-h; needs 80GB |
| GRPO-8B | Qwen/Qwen3-8B-Base | 3 | Replication check |

### 6c. Ablations (§6) — run in parallel on separate GPUs

All ablations use **Qwen3-4B-Base** to keep compute manageable.

| Ablation | Sweep values | GPUs needed | GPU-h est. |
|----------|-------------|-------------|------------|
| **A2 — Diversity metric** (most critical) | ① none ② 4-gram ③ semantic cls (DARLING) ④ embed full solution ⑤ embed trace only (COLD) | 5 | ~70h |
| A1 — Conditioning mechanism | ① no prefix ② random id ③ `#k` only ④ strategy prefix ⑤ COLD full | 5 | ~70h |
| A3 — Reward fusion | ① additive ② multiplicative ③ COLD gated | 3 | ~42h |
| A4 — K rollouts | K ∈ {2, 4, 6, 8, 12} | 5 | ~60h |
| A5 — λ sweep | λ ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 1.0} | 6 | ~84h |
| A7 — λ schedule | ① fixed ② step at epoch 3 ③ COLD linear ramp | 3 | ~42h |
| A8 — Reward normalization | with/without std-norm | 2 | ~28h |

### 6d. Science & instruction domains (§5.4–5.5)

| ID | Model | Domain | Notes |
|----|-------|--------|-------|
| COLD-SCI | Qwen3-8B-Base | SciKnowEval Chem+Physics | 5h wall-clock target |
| GRPO-SCI | Qwen3-8B-Base | Same | Comparison baseline |
| COLD-LLM | Llama-3.1-8B-Instruct | AlpacaEval / ArenaHard | WildChat 10k data |
| GRPO-LLM | Llama-3.1-8B-Instruct | Same | Comparison baseline |

---

## 7. Exact launch commands per experiment

### Primary 4-job parallel launcher (use this first)

```bash
cd COLD-RL
bash restart2.sh
```

This runs:
- GPU 0: COLD λ=0.5 (primary claim)
- GPU 1: COLD λ=0.1
- GPU 2: COLD λ=0.7
- GPU 3: GRPO λ=0.0 (baseline)

All with: `Qwen2.5-7B-Instruct`, DeepScaleR 10k, 1000 steps, LoRA rank 64, `save_every=100`.

### Manual single-job launch

```bash
# COLD λ=0.5 (primary)
CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python3 -u -m rl.run_rl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset deepscaler \
    --benchmark math \
    --n_rollouts 8 \
    --total_steps 1000 \
    --n_problems_per_step 32 \
    --mini_batch 8 \
    --ref_batch_size 4 \
    --max_new_tokens 4096 \
    --lr 1e-6 \
    --kl_beta 0.001 \
    --embed_model local \
    --lora_r 64 \
    --log_every 10 \
    --save_every 100 \
    --lambda_div 0.5 \
    --output_dir results/rl_runs/cold_deepscaler_lambda05_7b \
    > logs/rl_opt_lam05.log 2>&1 &

# GRPO baseline (λ=0)
CUDA_VISIBLE_DEVICES=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
  python3 -u -m rl.run_rl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset deepscaler \
    --benchmark math \
    --lambda_div 0.0 \
    --output_dir results/rl_runs/grpo_baseline_deepscaler_7b \
    > logs/rl_opt_grpo.log 2>&1 &
```

### 80 GB GPU — faster settings

On 80 GB cards you can raise batch sizes:

```bash
--ref_batch_size 8 --mini_batch 16
```

---

## 8. Monitoring

```bash
# Are jobs alive?
pgrep -af rl.run_rl

# GPU memory
nvidia-smi

# Live training metrics (step, loss, pass@1, diversity)
tail -f logs/rl_opt_lam05.log

# Structured metrics log (JSON per step)
tail -n 5 results/rl_runs/cold_deepscaler_lambda05_7b/training_log.jsonl

# Watch all four logs side by side
tail -f logs/rl_opt_lam05.log logs/rl_opt_lam01.log logs/rl_opt_lam07.log logs/rl_opt_grpo.log
```

**Evals** run automatically at `save_every=100` steps, reporting `pass@1`, `pass@4`, `pass@8`, `pass@16` via the COLD methodology (8 method approaches × 2 cycles = 16 samples per problem, 50 eval problems).

---

## 9. Memory settings by GPU

| GPU VRAM | `ref_batch_size` | `mini_batch` | `max_new_tokens` | Notes |
|----------|-----------------|-------------|-----------------|-------|
| 48 GB | 2–4 | 8 | 2048–4096 | vLLM re-init each step |
| 80 GB | 8 | 16 | 4096–8192 | Can keep vLLM persistent |
| 2× GPU | 8 | 16 | 8192 | Use two separate jobs, one per GPU |

---

## 10. Resuming from checkpoints

```bash
# Find latest checkpoint
ls results/rl_runs/cold_deepscaler_lambda05_7b/step_*/

# Resume
python3 -u -m rl.run_rl \
    ... \
    --resume_from results/rl_runs/cold_deepscaler_lambda05_7b/step_00200
```

`restart2.sh` resumes from `step_00100` by default. Update the `--resume_from` paths in that script to the latest checkpoint as training progresses.

---

## 11. If jobs die — recovery

```bash
# Kill any orphaned vLLM workers
pkill -9 -f run_rl
pkill -9 -f EngineCore
sleep 10

# Check GPU is clear
nvidia-smi

# Relaunch
bash restart2.sh
```

The init retry logic in `create_vllm_engine` (up to 6 attempts, backing off `gpu_memory_utilization` by ×0.88 each time) handles most vLLM OOM failures automatically without needing a manual restart.

---

## 12. Output directory structure

```
results/rl_runs/
  cold_deepscaler_lambda05_7b/
    step_00100/           # LoRA adapter checkpoint
      adapter_config.json
      adapter_model.safetensors
      metrics.json        # training + eval metrics at this step
    step_00200/
    training_log.jsonl    # one JSON line per log_every steps
    .vllm_lora_gen/       # temp LoRA weights for vLLM (overwritten each step)
```

---

## 13. Key hyperparameters reference

| Param | Current value | Paper target (Qwen3 runs) | What it controls |
|-------|--------------|--------------------------|-----------------|
| `model` | Qwen2.5-7B-Instruct | Qwen3-4B-Base / 8B-Base | Base model |
| `lambda_div` | 0.5 | 0.5 (ramp 0→0.5 over epochs 1–3) | Diversity reward weight |
| `n_rollouts` | 8 | 8 | Method prompts per problem |
| `temperature` | 0.8 | **Train 1.0 / Eval 0.6** | Rollout diversity |
| `max_new_tokens` | 4096 | 8192 (DARLING) | Max response length |
| `lr` | 1e-6 | 1e-6 | Learning rate |
| `kl_beta` | 0.001 | 0.0 for math, 0.001 for instruct | KL penalty weight |
| `lora_r` | 64 | 64 | LoRA rank |
| `n_problems_per_step` | 32 | 32 | Batch size in problems |
| `total_steps` | 1000 | ~1000 (10 epochs) | Training length |
| `embed_model` | local | local (e5-small-v2) | Diversity embedding |

---

## 14. Parallel execution plan for the paper (critical path)

```
Week 1 — 4 parallel GPUs minimum
  GPU 0: COLD λ=0.5  (primary result)
  GPU 1: COLD λ=0.1  (ablation A5)
  GPU 2: COLD λ=0.7  (ablation A5)
  GPU 3: GRPO λ=0.0  (baseline)

Week 1–2 — when more GPUs available
  GPU 4: COLD-4B (Qwen3-4B-Base)   ← paper model
  GPU 5: GRPO-4B (Qwen3-4B-Base)   ← replication check for DARLING
  GPU 6: COLD-8B (Qwen3-8B-Base)   ← scale
  GPU 7: GRPO-8B (Qwen3-8B-Base)   ← replication check

Week 2–3 — ablations (need 5–6 GPUs each)
  Ablation A2 (diversity metric):  5 GPU-parallel runs × ~14h = ~14h wall-clock
  Ablation A1 (conditioning):      5 runs × ~14h
  Ablation A5 (λ sweep):           6 runs × ~14h

Week 3 — domains
  COLD-SCI (Qwen3-8B, SciKnowEval): 5h wall-clock
  COLD-LLM (Llama-3.1-8B, WildChat): ~10h
```

**Minimum compute for a submittable paper:** ~115 GPU-hours of new training runs (critical §5 results only). With 4 GPUs running in parallel, that's **~29 hours wall-clock** before you have complete critical results.

---

## 15. What "SoTA" we are targeting

| Domain | Published SoTA (baseline) | Our target |
|--------|--------------------------|------------|
| Math (AIME25) | DARLING / DeepSeek-R1 | COLD-4B/8B pass@16 >> GRPO pass@16 |
| Math (OlympiadBench) | DARLING | COLD pass@1 ≥ GRPO; COLD pass@16 >> GRPO |
| Science (SciKnowEval) | SDPO avg@16 | COLD-SCI >> GRPO-SCI at 5h wall-clock |
| Instruction following | DARLING (AlpacaEval) | COLD-LLM win-rate > GRPO-LLM, ≥ DARLING |

The central claim is **not raw pass@1 beats everything**, it is that:  
> COLD significantly improves **pass@k for k > 1** (coverage / diversity) while maintaining or improving **pass@1** — something GRPO cannot do by design.
