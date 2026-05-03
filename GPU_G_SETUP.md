# GPU-G Setup — COLD-RL G-Track

Everything you need to run G1→G10 on the second A100 SXM 80GB.
GPU-A (the other machine) is already running the A-track automatically in tmux.

Both machines are **identical**: A100 SXM 80GB, same CUDA stack, same package versions.

---

## 1. Clone and install

```bash
git clone https://github.com/dude123studios/COLD-RL.git
cd COLD-RL
python3 -m venv .venv
source .venv/bin/activate

# PyTorch — match the CUDA version on this machine (check: nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Confirmed-working versions (same as GPU-A)
pip install \
  vllm==0.20.0 \
  transformers==5.7.0 \
  peft==0.19.1 \
  accelerate==1.13.0 \
  datasets \
  sentence-transformers \
  bitsandbytes \
  numpy \
  scipy
```

> **If CUDA version differs**: replace `cu124` with your version (e.g. `cu121`, `cu126`).
> Check with `nvidia-smi`, then pick the matching wheel at pytorch.org.

---

## 2. Pre-download models and data

Run these **before** starting the queue so it never stalls on a download mid-run:

```bash
source .venv/bin/activate
cd /workspace/COLD-RL

python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B', trust_remote_code=True)
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B', trust_remote_code=True)
print('Qwen2.5-7B OK')
"

python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-7B', trust_remote_code=True)
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Math-7B', trust_remote_code=True)
print('Qwen2.5-Math-7B OK')
"

python3 -c "
from datasets import load_dataset
ds = load_dataset('agentica-org/DeepScaleR-Preview-Dataset', split='train')
print(f'DeepScaleR OK — {len(ds)} problems')
"

python3 -c "
from datasets import load_dataset
ds = load_dataset('lighteval/MATH', split='train', trust_remote_code=True)
print(f'MATH train OK — {len(ds)} problems')
"

python3 -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('intfloat/e5-large-instruct', device='cpu')
print('E5 encoder OK')
"
```

---

## 3. Environment variables

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
```

Add to `~/.bashrc` or `~/.zshrc` so they survive reconnects.

---

## 4. Create output directories

```bash
mkdir -p /workspace/COLD-RL/logs /workspace/COLD-RL/results/rl_runs
```

---

## 5. Launch the queue (survives SSH disconnect)

```bash
cd /workspace/COLD-RL
source .venv/bin/activate
tmux new-session -d -s g_track 'bash /workspace/COLD-RL/run_gh200_queue.sh'
```

That's it. Runs G1 → G2 → G3 → G4 → G5 → G7 → G8 → G9 → G10 sequentially and logs everything to `logs/`.

---

## 6. Monitor

```bash
# Reattach to live session
tmux attach -t g_track

# Detach without killing: Ctrl+B then D

# Queue log
tail -f /workspace/COLD-RL/logs/gh200_queue.log

# Individual run log
tail -f /workspace/COLD-RL/logs/g3_cold_rl_qwen25_7b_s1.log

# GPU
watch -n 5 nvidia-smi
```

---

## 7. What the queue runs

| Job | Model | Algorithm | ~Hours | Purpose |
|-----|-------|-----------|--------|---------|
| G1 | Qwen2.5-7B | Zero-shot diagnostic | 2h | GO/NO-GO gate (see §8) |
| G2 | Qwen2.5-7B | GRPO baseline (λ=0) | 20h | Critical baseline |
| G3 | Qwen2.5-7B | COLD-RL seed 42 | 20h | **Primary result** |
| G4 | Qwen2.5-7B | COLD-RL seed 1337 | 20h | Seed 2 |
| G5 | Qwen2.5-7B | COLD-RL seed 0 | 20h | Seed 3 |
| G7 | Qwen2.5-Math-7B | COLD-RL seed 42 | 20h | Power Sampling comparison |
| G8 | Qwen2.5-7B | GRPO 12k steps | 30h | Compute-control ablation |
| G9 | Qwen2.5-7B | COLD-RL k∈{1,4,16} | 20h | Sparse k-schedule ablation |
| G10 | Qwen2.5-Math-7B | GRPO baseline (λ=0) | 20h | G7 reference baseline |

**Total: ~170h sequential.** The queue is `set -e` so a crash stops it — check the log and resume from checkpoint (§9).

G6 (ModC reproduction) is **not in this queue** — it requires a separate SFT script.

---

## 8. G1 diagnostic — what to look for

After G1 (~2h), check:

```bash
cat /workspace/COLD-RL/results/g1_diagnostic.json
```

| Field | Meaning |
|-------|---------|
| `delta_div_AminusB` | How much more diverse the prefix arm is vs baseline |
| `arm_a_prefix_tau07.pass_at_k` | pass@8 with prefix |
| `arm_b_nopfx_tau07.pass_at_k` | pass@8 without prefix |
| `verdict` | `"GO"` or `"NO-GO"` |

**GO**: `delta_div_AminusB > 0.04` AND prefix pass@8 ≥ baseline pass@8. Queue continues.

**NO-GO**: prefix adds no diversity signal. Kill the queue (`tmux kill-session -t g_track`), then post the diagnostic JSON so we can revise the prefix wording before restarting.

---

## 9. Resuming from a checkpoint

If a run crashes, find the last saved checkpoint:

```bash
ls results/rl_runs/g3_cold_rl_qwen25_7b_s1/
# e.g. step_04000/
```

Resume with:

```bash
python3 -m rl.run_rl \
    --model Qwen/Qwen2.5-7B \
    --dataset deepscaler --benchmark math \
    --n_rollouts 8 --temperature 1.0 --total_steps 8000 \
    --n_problems_per_step 32 --mini_batch 8 --ref_batch_size 4 \
    --max_new_tokens 4096 --lr 2e-6 --kl_beta 0.04 \
    --embed_model local --lora_r 32 --lora_alpha 64 \
    --phase1_steps 4000 --alpha_diversity 0.3 --k_max_training 16 \
    --seed 42 --gpu_id 0 \
    --resume_from results/rl_runs/g3_cold_rl_qwen25_7b_s1/step_04000 \
    --output_dir results/rl_runs/g3_cold_rl_qwen25_7b_s1
```

---

## 10. Running eval after a run completes

```bash
python3 -m evaluation.passk_method_eval \
    --model Qwen/Qwen2.5-7B \
    --lora-path results/rl_runs/g3_cold_rl_qwen25_7b_s1/step_08000 \
    --benchmarks math500 aime24 \
    --max-k 32 \
    --experiment-id g3_cold_rl_s1 \
    --out results/passk/g3_cold_rl_s1.json
```

Runs both **METHOD** mode (`[PARALLEL SAMPLE i OF k]` prefix) and **STANDARD** mode (no prefix) and prints a pass@k comparison table.

---

## Prompt format (for reference)

No system message. Instruction appended to user turn. Identical to DARLING/DeepScaleR format + our prefix.

**Standard inference (no prefix):**
```
<|im_start|>user
{problem}

Please reason step by step, and put your final answer within \boxed{}.
<|im_end|>
<|im_start|>assistant
```

**COLD-RL inference (with prefix):**
```
<|im_start|>user
[PARALLEL SAMPLE {i} OF {k}]

{problem}

Please reason step by step, and put your final answer within \boxed{}.
<|im_end|>
<|im_start|>assistant
```
