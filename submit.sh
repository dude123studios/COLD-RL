#!/usr/bin/env bash
#SBATCH --job-name=cold-rl-train
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=/data/user_data/shivansg/cold-rl-%x-%j.out
#SBATCH --error=/data/user_data/shivansg/cold-rl-%x-%j.err
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@60

# ============================================================
# COLD-RL preemption-safe SLURM job
#
# Preemption flow:
#   1. SLURM sends SIGTERM to this script 60s before hard kill
#   2. Bash trap forwards SIGTERM to the Python training process
#   3. Python sets _STOP_EVENT, completes the current step,
#      saves a checkpoint (LoRA + optimizer + scheduler), prunes
#      old checkpoints (keeps 2), then calls sys.exit(0)
#   4. This script exits 0 → --requeue puts the job back in queue
#   5. On next allocation, run_rl.py auto-detects the latest
#      checkpoint in OUTPUT_DIR and resumes transparently
#
# Customise via environment variables before sbatch:
#   MODEL=Qwen/Qwen2.5-7B OUTPUT_DIR=/path/to/dir sbatch submit.sh
# ============================================================

# ── Configurable parameters (override via env before sbatch) ─────────────────
MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
DATASET="${DATASET:-numinamath}"
BENCHMARK="${BENCHMARK:-math}"
N_EPOCHS="${N_EPOCHS:-8}"
LR="${LR:-1e-5}"             # 1e-5 for 7B; 1e-4 for ≤3B
EMBED_MODEL="${EMBED_MODEL:-local}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/user_data/shivansg/cold_rl_runs/cold_rl_main}"
SEED="${SEED:-42}"
GRPO_BASELINE="${GRPO_BASELINE:-}"   # set to "--grpo_baseline" for λ=0 runs
# ─────────────────────────────────────────────────────────────────────────────

echo "[submit.sh] Job ${SLURM_JOB_ID} starting on $(hostname) at $(date)"
echo "[submit.sh] Node list: ${SLURM_JOB_NODELIST:-unknown}"
echo "[submit.sh] Output dir: ${OUTPUT_DIR}"

# ── Environment ──────────────────────────────────────────────────────────────
source /home/shivansg/miniconda/etc/profile.d/conda.sh
conda activate env

export PYTORCH_ALLOC_CONF=expandable_segments:True

export HF_HOME=/data/user_data/shivansg/.hf_cache
export HF_DATASETS_CACHE=/data/user_data/shivansg/.hf_cache/datasets
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# Write triton autotune cache to /data to avoid NFS-related hangs on job exit.
export TRITON_CACHE_DIR=/data/user_data/shivansg/triton_cache

mkdir -p "${OUTPUT_DIR}"

# SLURM copies this script to its spool dir, so BASH_SOURCE[0] won't point
# back to the repo. Prefer the REPO_DIR env var exported by submit_all_experiments.sh.
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${REPO_DIR}"

# ── Graceful SIGTERM handler ──────────────────────────────────────────────────
# SLURM sends SIGTERM to this script (B: = batch shell) 60s before SIGKILL.
# We forward it to Python so the training loop can checkpoint and exit cleanly.
PYTHON_PID=""

_graceful_exit() {
    echo "[submit.sh] SIGTERM received at $(date) — forwarding to Python (PID=${PYTHON_PID})"
    if [[ -n "${PYTHON_PID}" ]] && kill -0 "${PYTHON_PID}" 2>/dev/null; then
        kill -TERM "${PYTHON_PID}"
        # Wait up to 55s for Python to finish its current step and save checkpoint.
        # The remaining 5s gives SLURM headroom before the hard SIGKILL.
        local deadline=$(( $(date +%s) + 55 ))
        while kill -0 "${PYTHON_PID}" 2>/dev/null && [[ $(date +%s) -lt ${deadline} ]]; do
            sleep 2
        done
        if kill -0 "${PYTHON_PID}" 2>/dev/null; then
            echo "[submit.sh] Python did not exit in time — sending SIGKILL"
            kill -KILL "${PYTHON_PID}" 2>/dev/null || true
        fi
    fi
    echo "[submit.sh] Exiting 0 — SLURM --requeue will resubmit the job"
    exit 0
}

trap '_graceful_exit' SIGTERM SIGINT

# ── Launch training ───────────────────────────────────────────────────────────
# run_rl.py auto-detects the latest checkpoint in OUTPUT_DIR and resumes;
# no --resume_from flag needed — the script handles it transparently.
python3 -u -m rl.run_rl \
    --model         "${MODEL}" \
    --dataset       "${DATASET}" \
    --benchmark     "${BENCHMARK}" \
    --n_rollouts    16 \
    --temperature   1.0 \
    --n_epochs      "${N_EPOCHS}" \
    --n_problems_per_step 16 \
    --mini_batch    8 \
    --ref_batch_size 4 \
    --max_new_tokens 8192 \
    --lr            "${LR}" \
    --kl_beta       0.01 \
    --clip_eps      0.2 \
    --embed_model   "${EMBED_MODEL}" \
    --lora_r        64 \
    --lora_alpha    128 \
    --log_every     10 \
    --save_every    50 \
    --seed          "${SEED}" \
    --gpu_id        0 \
    --output_dir    "${OUTPUT_DIR}" \
    ${GRPO_BASELINE} &

PYTHON_PID=$!
echo "[submit.sh] Python training PID=${PYTHON_PID}"

wait "${PYTHON_PID}"
EXIT_CODE=$?

echo "[submit.sh] Python exited with code ${EXIT_CODE} at $(date)"
exit "${EXIT_CODE}"
