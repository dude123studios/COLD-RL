# COLD-RL — Setup & Experiment Guide

> **What this is:** COLD (Controllable On-demand Learned Diversity) trains a Diversity-GRPO RL policy so the model produces *K* **genuinely distinct** solution approaches on demand while staying correct.  
> At inference: standard prompt → best single answer. `"Approach #k of K"` prompts → *K* diverse parallel solutions. Diversity is **controllable, not always-on**.

---

## ⚠️ Model: Always Use the Base Model

**Use `Qwen/Qwen2.5-7B` — NOT `Qwen/Qwen2.5-7B-Instruct`.**

This distinction is non-negotiable. Here is why:

| | Base (`Qwen2.5-7B`) | Instruct (`Qwen2.5-7B-Instruct`) |
|---|---|---|
| Prior RLHF | None | Yes — chat/helpfulness tuning applied |
| Policy malleability | High — RL can freely reshape the distribution | Low — prior tuning fights the new signal |
| (i,k) conditioning headroom | Full — model learns role assignment from scratch | Reduced — instruct constraints partially override |
| Used by DARLING, DeepSeek-R1, etc. | ✅ Base | ❌ |
| Correct for this project | ✅ | ❌ |

Using the Instruct model means 10+ hours of RLHF already spent constraining the distribution is working against you. The COLD reward needs to freely reshape how the model responds to `"Approach #i of k"` prompts — that requires a base model.

**The GRPO baseline must also use the base model.** A clean comparison requires both runs — baseline (λ=0.0) and COLD (λ=0.5) — to start from the identical model checkpoint. Mixing base and instruct starting points invalidates all comparisons.

---

## 1. Hardware Requirements

| Config | Min VRAM | Recommended |
|--------|----------|-------------|
| 7B training (this project) | 48 GB | 80 GB (A100-SXM4-80GB) |
| Eval only | 24 GB | any A100/H100 |

Each run occupies one GPU. Parallel runs require one GPU each.

---

## 2. Fresh GPU Setup (complete, copy-paste ready)

### Step 1 — Verify hardware

```bash
nvidia-smi
# Confirm: ≥48GB VRAM, driver ≥525
python3 --version
# Confirm: Python 3.10, 3.11, or 3.12
```

### Step 2 — Clone repo

```bash
# If repo is private, use your PAT:
git clone https://github.com/dude123studios/COLD-RL.git
cd COLD-RL
```

### Step 3 — Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

Confirmed working versions (as of 2026-05-03):
```
torch==2.11.0+cu130
vllm==0.20.0
transformers==5.7.0
peft==0.19.1
```

If vLLM install fails or produces `EngineCoreClient` errors, pin:
```bash
pip install "vllm==0.8.5"
```

### Step 5 — Pre-download model weights

Do this **before** training starts to avoid download overhead at the first step. Uses ~15 GB disk.

```bash
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading Qwen/Qwen2.5-7B base model...')
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B')
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B')
print('Done. Weights cached at:', tok.vocab_size)
del m
"
```

Alternatively, use the HuggingFace CLI:
```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-7B
```

### Step 6 — Pre-download DeepScaleR dataset

```bash
python3 -c "
from datasets import load_dataset
ds = load_dataset('agentica-org/DeepScaleR-Preview-Dataset')
print('DeepScaleR loaded:', len(ds['train']), 'problems')
"
```

### Step 7 — Set environment variables

```bash
# Always set — prevents CUDA allocator fragmentation on long runs
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Optional: redirect HF cache if default disk is small
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache

# Not required if using --embed_model local (e5-small-v2, which is the default)
# Only set if switching to OpenRouter embeddings:
# export OPENROUTER_API_KEY="sk-or-..."
```

Add these to `~/.bashrc` or prepend to every launch command.

### Step 8 — Create output directories

```bash
mkdir -p logs results/rl_runs
```

### Step 9 — Sanity check (optional but recommended)

Runs 1 training step with 4 problems to confirm vLLM init, LoRA, and reward pipeline work:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python3 -m rl.run_rl \
  --model Qwen/Qwen2.5-7B \
  --dataset deepscaler \
  --benchmark math \
  --n_rollouts 4 \
  --total_steps 1 \
  --n_problems_per_step 4 \
  --mini_batch 4 \
  --ref_batch_size 2 \
  --max_new_tokens 512 \
  --lambda_div 0.5 \
  --embed_model local \
  --output_dir results/rl_runs/sanity_check \
  --gpu_id 0
# Should complete in ~3 minutes. If it finishes without OOM, setup is good.
```

---

## 3. How COLD Works (90-second theory)

### The (i, k) conditioning design

Every problem gets *k* parallel rollouts. Each rollout receives a distinct prefix **before generation starts**. All *k* run simultaneously with zero inter-sample communication.

```
Sample 1 of k: π_θ(y | x, i=1, k=8)  ← role: first solver
Sample 2 of k: π_θ(y | x, i=2, k=8)  ← role: second solver, must differ
...
Sample k of k: π_θ(y | x, i=k, k=8)  ← role: k-th solver, covers remainder
```

The prefix is explicit text — not a soft token:
```
System: "You are a brilliant mathematician...asked to solve using a specific solution approach."
User:   "Solve using mathematical approach #i of k (each approach should be conceptually
         distinct — e.g., algebraic, geometric, number-theoretic, direct computation...)
         ...Approach #i: Solve step by step and conclude with \boxed{answer}."
```

This is the correct design because: all *k* samples run in parallel (no sequential dependency, no latency scaling with k), and each sample knows its role index *i* in the collective, not just the group size *k*. Conditioning on *k* alone pushes all samples toward higher entropy uniformly — that's just temperature scaling. Conditioning on *(i, k)* assigns each sample a distinct region of solution space.

### Reward

```
r_i = r_correct_i  +  λ × r_correct_i × diversity_score_i
```

- `r_correct_i`: binary (1 if math-verify passes, 0 if not)
- `diversity_score_i`: max cosine distance between any step in trace *i* vs any step in any other trace in the same group (via e5-small-v2 embeddings of reasoning steps)
- **Gated**: diversity bonus is zero if the answer is wrong — can't earn diversity by being creatively incorrect

### What λ controls

| λ | Effect |
|---|--------|
| 0.0 | Pure GRPO — no diversity signal **(clean baseline)** |
| 0.1 | Mild diversity nudge (ablation) |
| 0.5 | **Primary setting** — balanced correctness + diversity |
| 0.7 | Strong diversity pressure (ablation) |

### Why COLD beats DARLING

DARLING trains unconditional diversity — the model is always diverse, which can hurt accuracy on standard prompts. COLD trains *on-demand* diversity: same checkpoint, different inference-time behavior depending on whether you use method prompts.

---

## 4. Experiment Runs

### Priority order (with 1–2 GPUs available)

| Priority | Run | λ | GPU | Why |
|----------|-----|---|-----|-----|
| 1 | COLD λ=0.5 | 0.5 | new GPU | Primary paper result |
| 2 | GRPO baseline | 0.0 | current GPU | Must finish for comparison |
| 3 | COLD λ=0.7 | 0.7 | 3rd GPU if available | Ablation |
| 4 | COLD λ=0.1 | 0.1 | 4th GPU if available | Ablation |

### Exact launch command (all runs use identical flags, only `--lambda_div` and `--output_dir` differ)

**GRPO baseline (λ=0.0):**
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
nohup python3 -u -m rl.run_rl \
  --model Qwen/Qwen2.5-7B \
  --dataset deepscaler \
  --benchmark math \
  --n_rollouts 8 \
  --temperature 0.8 \
  --total_steps 700 \
  --n_problems_per_step 32 \
  --mini_batch 16 \
  --ref_batch_size 4 \
  --max_new_tokens 4096 \
  --lr 1e-6 \
  --kl_beta 0.001 \
  --embed_model local \
  --lora_r 64 \
  --lora_alpha 128 \
  --log_every 10 \
  --save_every 200 \
  --lambda_div 0.0 \
  --output_dir results/rl_runs/grpo_baseline_7b \
  --gpu_id 0 \
> logs/train_grpo_baseline_7b.log 2>&1 &
```

**COLD λ=0.5 (primary result) — run on new GPU:**
```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
nohup python3 -u -m rl.run_rl \
  --model Qwen/Qwen2.5-7B \
  --dataset deepscaler \
  --benchmark math \
  --n_rollouts 8 \
  --temperature 0.8 \
  --total_steps 700 \
  --n_problems_per_step 32 \
  --mini_batch 16 \
  --ref_batch_size 4 \
  --max_new_tokens 4096 \
  --lr 1e-6 \
  --kl_beta 0.001 \
  --embed_model local \
  --lora_r 64 \
  --lora_alpha 128 \
  --log_every 10 \
  --save_every 200 \
  --lambda_div 0.5 \
  --output_dir results/rl_runs/cold_deepscaler_lambda05_7b \
  --gpu_id 0 \
> logs/cold_lambda05_7b.log 2>&1 &
```

**COLD λ=0.7 (ablation):**  
Same as above, `--lambda_div 0.7 --output_dir results/rl_runs/cold_deepscaler_lambda070_7b > logs/cold_lambda07_7b.log`

**COLD λ=0.1 (ablation):**  
Same as above, `--lambda_div 0.1 --output_dir results/rl_runs/cold_deepscaler_lambda010_7b > logs/cold_lambda01_7b.log`

> **Clean comparison rule:** All runs must use the identical set of flags above — same model, same hyperparameters, same total_steps. The only thing that varies is `--lambda_div` and `--output_dir`. A baseline trained with different hyperparameters cannot be compared to a COLD run.

---

## 5. Monitoring

```bash
# Check jobs are alive
pgrep -af rl.run_rl

# GPU memory
nvidia-smi

# Live log
tail -f logs/cold_lambda05_7b.log

# Step-level metrics (loss, pass@1, diversity) as JSON
tail -n 5 results/rl_runs/cold_deepscaler_lambda05_7b/training_log.jsonl

# Watch multiple logs side by side
tail -f logs/train_grpo_baseline_7b.log logs/cold_lambda05_7b.log
```

Expected speed: **~11–13 minutes per step** on A100-80GB. First checkpoint at step 200 (~40h from cold start, ~25h from step 50).

---

## 6. Resuming from a Checkpoint

```bash
# Find latest checkpoint
ls results/rl_runs/cold_deepscaler_lambda05_7b/

# Resume (add --resume_from to any launch command above)
... --resume_from results/rl_runs/cold_deepscaler_lambda05_7b/step_00200 ...
```

---

## 7. Recovery After Job Death

```bash
pkill -9 -f run_rl
pkill -9 -f EngineCore
sleep 10
nvidia-smi   # confirm GPU memory is free
# Re-run the launch command above with --resume_from latest checkpoint
```

vLLM init retries automatically (up to 6 attempts, backing off `gpu_memory_utilization` ×0.88 each attempt) — most OOM failures self-heal without manual restart.

---

## 8. Output Structure

```
results/rl_runs/
  cold_deepscaler_lambda05_7b/
    step_00200/
      adapter_config.json
      adapter_model.safetensors
      metrics.json           # training + eval metrics at this checkpoint
    training_log.jsonl       # one JSON line per log_every steps
    .vllm_lora_gen/          # temp LoRA for vLLM, overwritten each step
```

---

## 9. Evaluation Protocol (pass@k)

Evals run automatically at each checkpoint. Metrics reported:
- `pass@1`, `pass@4`, `pass@8`, `pass@16`
- Method: 8 approach prompts × 2 cycles = 16 samples per problem, 50 eval problems
- **Do not evaluate pass@1 alone** — the whole point of COLD is pass@k for k > 1

---

## 10. Hyperparameter Reference

| Param | Value | Notes |
|-------|-------|-------|
| `model` | `Qwen/Qwen2.5-7B` | **Base model — no -Instruct suffix** |
| `lambda_div` | 0.0 / 0.5 / 0.7 / 0.1 | Varies per run — see §4 |
| `n_rollouts` | 8 | One method prompt per rollout |
| `temperature` | 0.8 | Training rollout temperature |
| `total_steps` | 700 | ~140h per run on A100-80GB |
| `n_problems_per_step` | 32 | Batch size in problems |
| `mini_batch` | 16 | Gradient mini-batch |
| `ref_batch_size` | 4 | Ref log-prob batch (reduce to 2 if OOM) |
| `max_new_tokens` | 4096 | Max response length |
| `lr` | 1e-6 | Learning rate |
| `kl_beta` | 0.001 | KL penalty |
| `lora_r` | 64 | LoRA rank |
| `lora_alpha` | 128 | LoRA alpha |
| `embed_model` | `local` | e5-small-v2, frozen, diversity scoring |
| `save_every` | 200 | Checkpoint interval |
| `log_every` | 10 | Metrics log interval |
