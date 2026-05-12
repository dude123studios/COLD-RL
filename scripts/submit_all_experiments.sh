#!/usr/bin/env bash
# ============================================================
# COLD-RL full experiment chain — E1 through E5 + baselines
#
# Design notes:
#   - BASE_DIR is on /data (compute nodes only). Login node never
#     mkdir's there — each job creates its own subdirs on startup.
#   - SLURM --output goes to HOME (NFS, writable everywhere).
#   - Eval/calib jobs are chained with afterok on training jobs.
#   - Water-filling is CPU-only → general partition.
#
# Usage:
#   bash scripts/submit_all_experiments.sh
# ============================================================

set -euo pipefail

BASE_DIR="/data/user_data/shivansg/cold_rl_runs"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Login-node writable dir for sbatch stdout/stderr and this script's log
SLOG="${HOME}/cold_rl_submit_logs"
mkdir -p "${SLOG}"
SUBMIT_LOG="${SLOG}/submit_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${SUBMIT_LOG}"; }

CONDA_INIT="source /home/shivansg/miniconda/etc/profile.d/conda.sh && conda activate env"
HF_EXPORTS="export HF_HOME=/data/user_data/shivansg/.hf_cache HF_DATASETS_CACHE=/data/user_data/shivansg/.hf_cache/datasets HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR=/data/user_data/shivansg/triton_cache"

# ---------------------------------------------------------------------------
# submit_setup: download all models + datasets to /data HF cache
# ---------------------------------------------------------------------------
submit_setup() {
    sbatch --parsable \
        --job-name="cold-setup" \
        --partition=cpu \
        --cpus-per-task=8 \
        --mem=32G \
        --time=12:00:00 \
        --output="${SLOG}/cold-setup-%j.out" \
        --error="${SLOG}/cold-setup-%j.err" \
        --wrap="
set -e
${CONDA_INIT}
export HF_HOME=/data/user_data/shivansg/.hf_cache
export HF_DATASETS_CACHE=/data/user_data/shivansg/.hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
mkdir -p \${HF_HOME}
cd ${REPO_DIR}
echo '[setup] Downloading models and datasets to /data HF cache...'
python3 - <<'PYEOF'
import os, sys
os.environ.setdefault('HF_HOME', '/data/user_data/shivansg/.hf_cache')
os.environ.setdefault('HF_DATASETS_CACHE', '/data/user_data/shivansg/.hf_cache/datasets')
from huggingface_hub import snapshot_download
from datasets import load_dataset

models = [
    'Qwen/Qwen2.5-7B',
    'Qwen/Qwen2.5-3B',
    'Qwen/Qwen2.5-1.5B',
    'Qwen/Qwen2.5-0.5B',
    'Qwen/Qwen2.5-7B-Instruct',
    'intfloat/e5-small-v2',
]
for m in models:
    print(f'[setup] Downloading model {m} ...', flush=True)
    snapshot_download(m)
    print(f'[setup]   Done: {m}', flush=True)

datasets_list = [
    ('AI-MO/NuminaMath-CoT', 'train', None),
    ('open-thoughts/OpenThoughts-3', 'train', None),
    ('qwedsacf/competition_math', 'train', None),
    ('lighteval/MATH', 'train', None),
    ('lighteval/MATH', 'test', None),
]
for name, split, cfg in datasets_list:
    print(f'[setup] Downloading dataset {name}:{split} ...', flush=True)
    kw = {'split': split}
    if cfg:
        kw['name'] = cfg
    try:
        load_dataset(name, **kw)
        print(f'[setup]   Done: {name}:{split}', flush=True)
    except Exception as e:
        print(f'[setup]   ERROR {name}:{split}: {e}', flush=True)
        sys.exit(1)
print('[setup] All downloads complete.', flush=True)
PYEOF
echo '[setup] Running benchmark download...'
python3 -m scripts.download_benchmarks
echo '[setup] Setup complete.'
"
}

# ---------------------------------------------------------------------------
# submit_train: submit a training job, echo back the SLURM job ID
# ---------------------------------------------------------------------------
submit_train() {
    local name="$1" model="$2" lr="$3" dep_jid="$4"
    shift 4
    local extra_env="${*:-}"   # optional "KEY=VALUE ..." to append to --export

    local export_str="ALL,REPO_DIR=${REPO_DIR},MODEL=${model},LR=${lr},OUTPUT_DIR=${BASE_DIR}/${name}"
    [[ -n "${extra_env}" ]] && export_str="${export_str},${extra_env}"

    local dep_flag=""
    [[ -n "${dep_jid}" ]] && dep_flag="--dependency=afterok:${dep_jid}"

    sbatch --parsable \
        --job-name="cold-${name}" \
        --export="${export_str}" \
        --output="${SLOG}/cold-${name}-%j.out" \
        --error="${SLOG}/cold-${name}-%j.err" \
        ${dep_flag} \
        "${REPO_DIR}/submit.sh"
}

# ---------------------------------------------------------------------------
# submit_eval: GPU eval job depending on a training job
# ---------------------------------------------------------------------------
submit_eval() {
    local name="$1" dep_jid="$2" model="$3" lora_dir="$4" benchmark="$5" maxk="${6:-64}"

    local out_json="${BASE_DIR}/eval/${name}.json"

    sbatch --parsable \
        --job-name="eval-${name}" \
        --partition=preempt \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=60G \
        --time=4:00:00 \
        --requeue \
        --output="${SLOG}/eval-${name}-%j.out" \
        --dependency="afterok:${dep_jid}" \
        --wrap="
set -e
${CONDA_INIT}
${HF_EXPORTS}
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p ${BASE_DIR}/eval
cd ${REPO_DIR}
python3 -m evaluation.passk_method_eval \
    --model ${model} \
    --lora-path ${lora_dir} \
    --benchmarks ${benchmark} \
    --max-k ${maxk} \
    --experiment-id ${name} \
    --out ${out_json}
"
}

# ---------------------------------------------------------------------------
# submit_calib: calibration job (spec §2.1: 500 problems × 64 draws × 16 indices)
# ---------------------------------------------------------------------------
submit_calib() {
    local name="$1" dep_jid="$2" model="$3" lora_dir="$4" benchmark="$5"

    local out_json="${BASE_DIR}/calibration/${name}.json"

    sbatch --parsable \
        --job-name="calib-${name}" \
        --partition=preempt \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=60G \
        --time=12:00:00 \
        --requeue \
        --output="${SLOG}/calib-${name}-%j.out" \
        --dependency="afterok:${dep_jid}" \
        --wrap="
set -e
${CONDA_INIT}
${HF_EXPORTS}
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p ${BASE_DIR}/calibration
cd ${REPO_DIR}
python3 -m evaluation.calibration \
    --model ${model} \
    --lora-path ${lora_dir} \
    --benchmark ${benchmark} \
    --n-problems 500 \
    --R 64 \
    --n-max 128 \
    --dataset math500 \
    --out ${out_json}
"
}

# ---------------------------------------------------------------------------
# submit_wf: CPU-only water-filling job depending on calibration
# ---------------------------------------------------------------------------
submit_wf() {
    local name="$1" dep_jid="$2" calib_json="$3"

    local out_json="${BASE_DIR}/water_filling/${name}.json"

    sbatch --parsable \
        --job-name="wf-${name}" \
        --partition=cpu \
        --cpus-per-task=4 \
        --mem=16G \
        --time=0:30:00 \
        --output="${SLOG}/wf-${name}-%j.out" \
        --dependency="afterok:${dep_jid}" \
        --wrap="
set -e
${CONDA_INIT}
mkdir -p ${BASE_DIR}/water_filling
cd ${REPO_DIR}
python3 -m evaluation.water_filling \
    --calib ${calib_json} \
    --k-values 1 2 4 8 16 32 64 128 256 512 \
    --out ${out_json}
"
}

# ===========================================================================
# Setup — download all models and datasets before any training starts
# ===========================================================================
log "=== Setup: downloading models + datasets ==="
JID_SETUP=$(submit_setup)
log "  setup → job ${JID_SETUP}"

# ===========================================================================
# E1 — Main pass@k sweep: Qwen2.5-Base 0.5B / 1.5B / 3B / 7B
# ===========================================================================
log "=== E1: Main pass@k sweep ==="

JID_E1_7B=$(submit_train   "e1_7b"    "Qwen/Qwen2.5-7B"    "1e-5"  "${JID_SETUP}")
log "  E1 7B   → job ${JID_E1_7B}"

JID_E1_3B=$(submit_train   "e1_3b"    "Qwen/Qwen2.5-3B"    "1e-4"  "${JID_SETUP}")
log "  E1 3B   → job ${JID_E1_3B}"

JID_E1_1P5B=$(submit_train "e1_1p5b"  "Qwen/Qwen2.5-1.5B"  "1e-4"  "${JID_SETUP}")
log "  E1 1.5B → job ${JID_E1_1P5B}"

JID_E1_0P5B=$(submit_train "e1_0p5b"  "Qwen/Qwen2.5-0.5B"  "1e-4"  "${JID_SETUP}")
log "  E1 0.5B → job ${JID_E1_0P5B}"

# Eval jobs (afterok on training; eval uses the output_dir as lora_dir — run_rl.py
# saves step_XXXXX/ inside it; passk_method_eval will use the highest-step checkpoint)
JID_EVAL_7B=$(submit_eval   "e1_7b"    "${JID_E1_7B}"    "Qwen/Qwen2.5-7B"   "${BASE_DIR}/e1_7b"    "math500" "128")
log "  eval 7B → job ${JID_EVAL_7B}"

JID_EVAL_3B=$(submit_eval   "e1_3b"    "${JID_E1_3B}"    "Qwen/Qwen2.5-3B"   "${BASE_DIR}/e1_3b"    "math500" "128")
log "  eval 3B → job ${JID_EVAL_3B}"

JID_EVAL_1P5B=$(submit_eval "e1_1p5b"  "${JID_E1_1P5B}"  "Qwen/Qwen2.5-1.5B" "${BASE_DIR}/e1_1p5b"  "math500" "128")
log "  eval 1.5B → job ${JID_EVAL_1P5B}"

JID_EVAL_0P5B=$(submit_eval "e1_0p5b"  "${JID_E1_0P5B}"  "Qwen/Qwen2.5-0.5B" "${BASE_DIR}/e1_0p5b"  "math500" "128")
log "  eval 0.5B → job ${JID_EVAL_0P5B}"

# ===========================================================================
# GRPO baseline (λ=0) — required for E1 comparison table and E5
# ===========================================================================
log "=== GRPO baseline (λ=0) ==="

JID_GRPO_7B=$(submit_train "grpo_baseline_7b" "Qwen/Qwen2.5-7B" "1e-5" "${JID_SETUP}" \
    "GRPO_BASELINE=--grpo_baseline")
log "  GRPO 7B → job ${JID_GRPO_7B}"

JID_EVAL_GRPO=$(submit_eval "grpo_baseline_7b" "${JID_GRPO_7B}" \
    "Qwen/Qwen2.5-7B" "${BASE_DIR}/grpo_baseline_7b" "math500" "128")
log "  eval GRPO 7B → job ${JID_EVAL_GRPO}"

# ===========================================================================
# E2 — Long-CoT benchmark (Qwen2.5-7B-Instruct, OpenThoughts-3)
# ===========================================================================
log "=== E2: Long-CoT benchmark ==="

JID_E2=$(submit_train "e2_longcot_7b" "Qwen/Qwen2.5-7B-Instruct" "1e-5" "${JID_SETUP}" \
    "DATASET=openthoughts3")
log "  E2 7B-Instruct → job ${JID_E2}"

JID_EVAL_E2=$(submit_eval "e2_aime" "${JID_E2}" \
    "Qwen/Qwen2.5-7B-Instruct" "${BASE_DIR}/e2_longcot_7b" "aime24" "64")
log "  eval E2 AIME → job ${JID_EVAL_E2}"

# ===========================================================================
# E3 — P(n,i) saturation curves (calibrate on 7B after E1 training)
# ===========================================================================
log "=== E3: P(n,i) saturation curves ==="

JID_CALIB_E3=$(submit_calib "e3_saturation" "${JID_E1_7B}" \
    "Qwen/Qwen2.5-7B" "${BASE_DIR}/e1_7b" "math")
log "  calib E3 → job ${JID_CALIB_E3}"

JID_WF_E3=$(submit_wf "e3_saturation" "${JID_CALIB_E3}" \
    "${BASE_DIR}/calibration/e3_saturation.json")
log "  water-fill E3 → job ${JID_WF_E3}"

# ===========================================================================
# E4 — Allocator comparison (fresh out-of-sample 500-problem calibration)
# ===========================================================================
log "=== E4: Allocator comparison ==="

JID_CALIB_E4=$(submit_calib "e4_oos" "${JID_E1_7B}" \
    "Qwen/Qwen2.5-7B" "${BASE_DIR}/e1_7b" "math")
log "  calib E4 → job ${JID_CALIB_E4}"

JID_WF_E4=$(submit_wf "e4_oos" "${JID_CALIB_E4}" \
    "${BASE_DIR}/calibration/e4_oos.json")
log "  water-fill E4 → job ${JID_WF_E4}"

# ===========================================================================
# E5 — Controllability check: COLD-RL without prefix vs GRPO
# ===========================================================================
log "=== E5: Controllability check ==="

JID_E5=$(sbatch --parsable \
    --job-name="e5-nopfx" \
    --partition=preempt \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=60G \
    --time=4:00:00 \
    --requeue \
    --output="${SLOG}/e5-nopfx-%j.out" \
    --dependency="afterok:${JID_E1_7B}" \
    --wrap="
set -e
${CONDA_INIT}
${HF_EXPORTS}
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p ${BASE_DIR}/eval
cd ${REPO_DIR}
python3 -m evaluation.passk_method_eval \
    --model Qwen/Qwen2.5-7B \
    --lora-path ${BASE_DIR}/e1_7b \
    --benchmarks math500 \
    --max-k 1 \
    --experiment-id e5_cold_noprefix \
    --out ${BASE_DIR}/eval/e5_cold_noprefix.json
")
log "  E5 no-prefix → job ${JID_E5}"

# ===========================================================================
# Summary
# ===========================================================================
log ""
log "══════════════════════════════════════════════"
log "Submission complete. All jobs queued."
log "══════════════════════════════════════════════"
log "Setup (must finish before training starts):"
log "  setup          ${JID_SETUP}"
log "Training:"
log "  e1_7b          ${JID_E1_7B}"
log "  e1_3b          ${JID_E1_3B}"
log "  e1_1p5b        ${JID_E1_1P5B}"
log "  e1_0p5b        ${JID_E1_0P5B}"
log "  grpo_baseline  ${JID_GRPO_7B}"
log "  e2_longcot     ${JID_E2}"
log "Monitor: squeue -u \$USER"
log "Logs:    ${SLOG}/"
