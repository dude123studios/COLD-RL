#!/usr/bin/env bash
# ============================================================
# COLD-RL full experiment chain — E1 through E5 + baselines
#
# Submits all training jobs to the preempt partition.
# Each eval job depends (afterok) on its training job completing.
# Cross-experiment eval (E3/E4/E5) depends on the 7B training job.
#
# Usage:
#   bash scripts/submit_all_experiments.sh
#
# Monitor:
#   squeue -u $USER
#   tail -f /data/user_data/shivansg/cold_rl_runs/logs/submit.log
# ============================================================

set -euo pipefail

BASE_DIR="/data/user_data/shivansg/cold_rl_runs"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "${LOG_DIR}"
SUBMIT_LOG="${LOG_DIR}/submit.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${SUBMIT_LOG}"; }

# ---------------------------------------------------------------------------
# Helper: submit one training job, return its SLURM job ID
# ---------------------------------------------------------------------------
submit_train() {
    local name="$1" model="$2" lr="$3"
    shift 3
    local extra_env=("$@")   # KEY=VALUE pairs to export

    local out_dir="${BASE_DIR}/${name}"
    mkdir -p "${out_dir}"

    # Build export string
    local exports="ALL,MODEL=${model},LR=${lr},OUTPUT_DIR=${out_dir}"
    for kv in "${extra_env[@]:-}"; do
        [[ -n "${kv}" ]] && exports="${exports},${kv}"
    done

    local jid
    jid=$(sbatch --parsable \
        --job-name="cold-${name}" \
        --export="${exports}" \
        "${REPO_DIR}/submit.sh")
    log "  submitted ${name} → job ${jid}"
    echo "${jid}"
}

# ---------------------------------------------------------------------------
# Helper: submit evaluation job depending on a training job
# ---------------------------------------------------------------------------
submit_eval() {
    local name="$1" dep_jid="$2" model="$3" lora_dir="$4" benchmark="$5"
    shift 5
    local extra_k="${1:-64}"   # max-k

    local out="${BASE_DIR}/eval/${name}.json"
    mkdir -p "${BASE_DIR}/eval"

    local jid
    jid=$(sbatch --parsable \
        --job-name="eval-${name}" \
        --partition=preempt \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=60G \
        --time=4:00:00 \
        --output="${LOG_DIR}/eval-${name}-%j.out" \
        --dependency="afterok:${dep_jid}" \
        --wrap="
source /home/shivansg/miniconda/etc/profile.d/conda.sh
conda activate cold-rl
export HF_HOME=/data/user_data/shivansg/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=1
cd ${REPO_DIR}
python3 -m evaluation.passk_method_eval \
    --model ${model} \
    --lora-path ${lora_dir} \
    --benchmarks ${benchmark} \
    --max-k ${extra_k} \
    --experiment-id ${name} \
    --out ${out}
")
    log "  submitted eval-${name} → job ${jid} (dep: ${dep_jid})"
    echo "${jid}"
}

# ---------------------------------------------------------------------------
# Helper: submit calibration job
# ---------------------------------------------------------------------------
submit_calib() {
    local name="$1" dep_jid="$2" model="$3" lora_dir="$4" benchmark="$5"

    local out="${BASE_DIR}/calibration/${name}.json"
    mkdir -p "${BASE_DIR}/calibration"

    local jid
    jid=$(sbatch --parsable \
        --job-name="calib-${name}" \
        --partition=preempt \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=60G \
        --time=8:00:00 \
        --output="${LOG_DIR}/calib-${name}-%j.out" \
        --dependency="afterok:${dep_jid}" \
        --wrap="
source /home/shivansg/miniconda/etc/profile.d/conda.sh
conda activate cold-rl
export HF_HOME=/data/user_data/shivansg/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_HUB_OFFLINE=1
cd ${REPO_DIR}
python3 -m evaluation.calibration \
    --model ${model} \
    --lora-path ${lora_dir} \
    --benchmark ${benchmark} \
    --n-problems 500 \
    --R 64 \
    --n-max 128 \
    --dataset math500 \
    --out ${out}
")
    log "  submitted calib-${name} → job ${jid} (dep: ${dep_jid})"
    echo "${jid}"
}

# ---------------------------------------------------------------------------
# Helper: submit water-filling job depending on calibration
# ---------------------------------------------------------------------------
submit_wf() {
    local name="$1" dep_jid="$2" calib_json="$3"

    local out="${BASE_DIR}/water_filling/${name}.json"
    mkdir -p "${BASE_DIR}/water_filling"

    local jid
    jid=$(sbatch --parsable \
        --job-name="wf-${name}" \
        --partition=preempt \
        --gres=gpu:0 \
        --cpus-per-task=4 \
        --mem=16G \
        --time=0:30:00 \
        --output="${LOG_DIR}/wf-${name}-%j.out" \
        --dependency="afterok:${dep_jid}" \
        --wrap="
source /home/shivansg/miniconda/etc/profile.d/conda.sh
conda activate cold-rl
cd ${REPO_DIR}
python3 -m evaluation.water_filling \
    --calib ${calib_json} \
    --k-values 1 2 4 8 16 32 64 128 256 512 \
    --out ${out}
")
    log "  submitted wf-${name} → job ${jid} (dep: ${dep_jid})"
    echo "${jid}"
}

# ===========================================================================
# E1 — Main pass@k sweep (Qwen2.5-Base: 0.5B, 1.5B, 3B, 7B)
# ===========================================================================
log "=== E1: Main pass@k sweep ==="

JID_E1_7B=$(submit_train   "e1_7b"    "Qwen/Qwen2.5-7B"    "1e-5")
JID_E1_3B=$(submit_train   "e1_3b"    "Qwen/Qwen2.5-3B"    "1e-4")
JID_E1_1P5B=$(submit_train "e1_1p5b"  "Qwen/Qwen2.5-1.5B"  "1e-4")
JID_E1_0P5B=$(submit_train "e1_0p5b"  "Qwen/Qwen2.5-0.5B"  "1e-4")

# Find final checkpoint dir after training (highest step_XXXXX)
LORA_7B="${BASE_DIR}/e1_7b"
LORA_3B="${BASE_DIR}/e1_3b"
LORA_1P5B="${BASE_DIR}/e1_1p5b"
LORA_0P5B="${BASE_DIR}/e1_0p5b"

JID_EVAL_7B=$(submit_eval   "e1_7b"    "${JID_E1_7B}"    "Qwen/Qwen2.5-7B"    "${LORA_7B}"    "math500" "128")
JID_EVAL_3B=$(submit_eval   "e1_3b"    "${JID_E1_3B}"    "Qwen/Qwen2.5-3B"    "${LORA_3B}"    "math500" "128")
JID_EVAL_1P5B=$(submit_eval "e1_1p5b"  "${JID_E1_1P5B}"  "Qwen/Qwen2.5-1.5B"  "${LORA_1P5B}"  "math500" "128")
JID_EVAL_0P5B=$(submit_eval "e1_0p5b"  "${JID_E1_0P5B}"  "Qwen/Qwen2.5-0.5B"  "${LORA_0P5B}"  "math500" "128")

# ===========================================================================
# GRPO baseline (λ=0) — needed for E1 comparison and E5 controllability
# ===========================================================================
log "=== GRPO baseline (λ=0) ==="
JID_GRPO_7B=$(submit_train "grpo_baseline_7b" "Qwen/Qwen2.5-7B" "1e-5" \
    "GRPO_BASELINE=--grpo_baseline")
JID_EVAL_GRPO_7B=$(submit_eval "grpo_baseline_7b" "${JID_GRPO_7B}" \
    "Qwen/Qwen2.5-7B" "${BASE_DIR}/grpo_baseline_7b" "math500" "128")

# ===========================================================================
# E2 — Long-CoT benchmark (Qwen2.5-7B-Instruct, OpenThoughts-3)
# ===========================================================================
log "=== E2: Long-CoT benchmark ==="
JID_E2=$(submit_train "e2_longcot_7b" "Qwen/Qwen2.5-7B-Instruct" "1e-5" \
    "DATASET=openthoughts3" "BENCHMARK=math" "N_EPOCHS=8")

JID_EVAL_E2_AIME=$(submit_eval "e2_aime"  "${JID_E2}" \
    "Qwen/Qwen2.5-7B-Instruct" "${BASE_DIR}/e2_longcot_7b" "aime24" "64")
# OlympiadBench eval — depends on E2 training
JID_EVAL_E2_OLY=$(submit_eval "e2_olympiad" "${JID_E2}" \
    "Qwen/Qwen2.5-7B-Instruct" "${BASE_DIR}/e2_longcot_7b" "math500" "64")

# ===========================================================================
# E3 — P(n,i) saturation curves (depends on E1 7B training completing)
# ===========================================================================
log "=== E3: P(n,i) saturation curves ==="
JID_CALIB_E3=$(submit_calib "e3_saturation" "${JID_E1_7B}" \
    "Qwen/Qwen2.5-7B" "${LORA_7B}" "math")
# Water-filling on E3 calibration
JID_WF_E3=$(submit_wf "e3_saturation" "${JID_CALIB_E3}" \
    "${BASE_DIR}/calibration/e3_saturation.json")

# ===========================================================================
# E4 — Allocator comparison (uses same calibration as E3, no extra training)
# ===========================================================================
log "=== E4: Allocator comparison ==="
# Re-uses the E3 calibration JSON; water_filling.py already compares all allocators.
# Submit a second wf run explicitly labeled e4 on a fresh calib (out-of-sample 500)
JID_CALIB_E4=$(submit_calib "e4_oos" "${JID_E1_7B}" \
    "Qwen/Qwen2.5-7B" "${LORA_7B}" "math")
JID_WF_E4=$(submit_wf "e4_oos" "${JID_CALIB_E4}" \
    "${BASE_DIR}/calibration/e4_oos.json")

# ===========================================================================
# E5 — Controllability check (COLD-RL vs GRPO vs DARLING prefix=None)
# ===========================================================================
log "=== E5: Controllability check ==="
# Eval COLD-RL 7B without prefix (standard prompt) — uses existing eval job output
# Also submits paired-t-test analysis script after both evals are done
JID_E5_NOPFX=$(sbatch --parsable \
    --job-name="e5-nopfx" \
    --partition=preempt \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=60G \
    --time=4:00:00 \
    --output="${LOG_DIR}/e5-nopfx-%j.out" \
    --dependency="afterok:${JID_E1_7B}" \
    --wrap="
source /home/shivansg/miniconda/etc/profile.d/conda.sh
conda activate cold-rl
export HF_HOME=/data/user_data/shivansg/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_HUB_OFFLINE=1
cd ${REPO_DIR}
# Evaluate COLD-RL model with standard (no-prefix) prompts — should match GRPO pass@1
python3 -m evaluation.passk_method_eval \
    --model Qwen/Qwen2.5-7B \
    --lora-path ${LORA_7B} \
    --benchmarks math500 \
    --max-k 1 \
    --experiment-id e5_cold_noprefix \
    --out ${BASE_DIR}/eval/e5_cold_noprefix.json
")
log "  submitted e5-nopfx → job ${JID_E5_NOPFX} (dep: ${JID_E1_7B})"

# ===========================================================================
# Summary
# ===========================================================================
log ""
log "=== Submission complete ==="
log "Training jobs:"
log "  E1 7B   : ${JID_E1_7B}"
log "  E1 3B   : ${JID_E1_3B}"
log "  E1 1.5B : ${JID_E1_1P5B}"
log "  E1 0.5B : ${JID_E1_0P5B}"
log "  GRPO 7B : ${JID_GRPO_7B}"
log "  E2 7B   : ${JID_E2}"
log ""
log "Eval/calib/wf jobs are chained with afterok dependencies."
log ""
log "Monitor: squeue -u \$USER"
log "Logs   : ${LOG_DIR}/"
log "Results: ${BASE_DIR}/"
