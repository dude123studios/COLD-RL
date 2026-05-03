# GH200 Setup — COLD-RL G-Track

Everything you need to run G1→G10 on the GH200 96GB GPU.
The A100 (this machine) is already running the A-track automatically.

---

## 1. Clone and install

```bash
git clone https://github.com/dude123studios/COLD-RL.git
cd COLD-RL
python3 -m venv .venv
source .venv/bin/activate

# PyTorch — use the CUDA version matching your driver (check: nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Core dependencies (versions confirmed working on A100 / should work on GH200)
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
> Run `nvidia-smi` to check driver, then pick the matching wheel from pytorch.org.

---

## 2. Pre-download models and data (do this before starting the queue)

Run these once so the queue never stalls waiting on a download:

```bash
# Models
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

# Datasets
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

# Embedding model (used for diversity reward)
python3 -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('intfloat/e5-large-instruct', device='cpu')
print('E5 encoder OK')
"
```

---

## 3. Set environment variables

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
```

Add these to `~/.bashrc` or `~/.zshrc` so they persist across SSH sessions.

---

## 4. Create output directories

```bash
mkdir -p /workspace/COLD-RL/logs /workspace/COLD-RL/results/rl_runs
```

---

## 5. Launch the queue (survives SSH disconnect)

```bash
cd /workspace/COLD-RL
tmux new-session -d -s gh200 'bash /workspace/COLD-RL/run_gh200_queue.sh'
```

That's it. The queue runs G1 → G2 → G3 → G4 → G5 → G7 → G8 → G9 → G10 sequentially and logs everything.

---

## 6. Monitor

```bash
# Reattach to the live session
tmux attach -t gh200

# Detach without killing: Ctrl+B then D

# Watch the combined queue log
tail -f /workspace/COLD-RL/logs/gh200_queue.log

# Watch a specific run
tail -f /workspace/COLD-RL/logs/g3_cold_rl_qwen25_7b_s1.log

# GPU utilisation
watch -n 5 nvidia-smi
```

---

## 7. What the queue runs

| Job | Model | Algorithm | Hours | Purpose |
|-----|-------|-----------|-------|---------|
| G1 | Qwen2.5-7B | Zero-shot diagnostic | ~2h | **GO/NO-GO gate** — checks prefix induces diversity before any training |
| G2 | Qwen2.5-7B | GRPO baseline (λ=0) | ~8h | Critical baseline |
| G3 | Qwen2.5-7B | COLD-RL seed 42 | ~8h | **Primary result** |
| G4 | Qwen2.5-7B | COLD-RL seed 1337 | ~8h | Seed 2 |
| G5 | Qwen2.5-7B | COLD-RL seed 0 | ~8h | Seed 3 |
| G7 | Qwen2.5-Math-7B | COLD-RL seed 42 | ~9h | Power Sampling comparison |
| G8 | Qwen2.5-7B | GRPO 12k steps | ~12h | Compute-control ablation |
| G9 | Qwen2.5-7B | COLD-RL k∈{1,4,16} | ~8h | Sparse k-schedule ablation |
| G10 | Qwen2.5-Math-7B | GRPO baseline (λ=0) | ~8h | G7 reference baseline |

G6 (ModC reproduction) is **not in the queue** — it requires a separate SFT script not yet implemented.

Total wall time: ~75h sequential.

---

## 8. G1 diagnostic — what to look for

After G1 finishes (~2h), check `results/g1_diagnostic.json`:

```bash
cat results/g1_diagnostic.json
```

**GO** if `delta_div_AminusB > 0.04` AND `arm_a pass@8 >= arm_b pass@8`.
The queue continues regardless — kill it only if you see a hard NO-GO and want to revise the prefix first.

---

## 9. Checkpoints

Checkpoints save every 1000 steps to `results/rl_runs/<run_name>/step_XXXXX/`.

To resume a run from a checkpoint if something crashes:

```bash
python3 -m rl.run_rl \
    --model Qwen/Qwen2.5-7B \
    --dataset deepscaler --benchmark math \
    --resume_from results/rl_runs/g3_cold_rl_qwen25_7b_s1/step_03000 \
    --output_dir results/rl_runs/g3_cold_rl_qwen25_7b_s1 \
    ... (same flags as original run)
```

---

## 10. After runs finish — run eval

Once G2 or G3 completes, run eval on its checkpoint:

```bash
python3 -m evaluation.passk_method_eval \
    --model Qwen/Qwen2.5-7B \
    --lora-path results/rl_runs/g3_cold_rl_qwen25_7b_s1/step_08000 \
    --benchmarks math500 aime24 \
    --max-k 32 \
    --experiment-id g3_cold_rl_s1 \
    --out results/passk/g3_cold_rl_s1.json
```

This runs both METHOD mode (with `[PARALLEL SAMPLE i OF k]` prefix) and STANDARD mode (no prefix) and prints a comparison table.

---

## Prompt format reference

All training and eval prompts follow this format exactly (no system message, instruction in user turn — matches DARLING/DeepScaleR):

**Standard (k=1, baseline, no prefix):**
```
<|im_start|>user
{problem}

Please reason step by step, and put your final answer within \boxed{}.
<|im_end|>
<|im_start|>assistant
```

**COLD-RL (k>1, diversity mode):**
```
<|im_start|>user
[PARALLEL SAMPLE {i} OF {k}]

{problem}

Please reason step by step, and put your final answer within \boxed{}.
<|im_end|>
<|im_start|>assistant
```

At inference: use **no prefix** for best single answer, use **`[PARALLEL SAMPLE i OF k]`** for k diverse parallel answers.
