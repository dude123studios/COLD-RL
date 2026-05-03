#!/usr/bin/env bash
# ============================================================
# GH200 96GB — Sequential training queue
# Runs: G1 -> G2 -> G3 -> G4 -> G5 -> G7 -> G8 -> G9 -> G10
#
# NOTE: G6 (ModC reproduction) requires a separate SFT script
#       not yet implemented. It will be skipped in this queue
#       and must be run manually from scripts/run_modc.py.
#
# LAUNCH (survives SSH disconnect):
#   tmux new-session -d -s gh200 'bash /workspace/COLD-RL/run_gh200_queue.sh'
# REATTACH:
#   tmux attach -t gh200
# MONITOR:
#   tail -f /workspace/COLD-RL/logs/gh200_queue.log
# ============================================================
set -e
cd /workspace/COLD-RL
mkdir -p logs results/rl_runs

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

LOG=/workspace/COLD-RL/logs/gh200_queue.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Shared flags for all RL training runs on GH200
COMMON=(
    --dataset deepscaler
    --benchmark math
    --n_rollouts 8
    --temperature 1.0
    --total_steps 8000
    --n_problems_per_step 32
    --mini_batch 8
    --ref_batch_size 4
    --max_new_tokens 4096
    --lr 2e-6
    --kl_beta 0.04
    --embed_model local
    --lora_r 32
    --lora_alpha 64
    --log_every 10
    --save_every 1000
    --phase1_steps 4000
    --alpha_diversity 0.3
    --k_max_training 16
    --gpu_id 0
)

run_job() {
    local name="$1"; shift
    log "=============================="
    log "Starting $name"
    log "=============================="
    python3 -u -m rl.run_rl "$@" 2>&1 | tee -a "logs/${name}.log"
    log "$name COMPLETE"
}

# ── G1: Zero-shot diagnostic (GO/NO-GO gate) ─────────────────────────────────
log "=============================="
log "G1: Zero-shot diagnostic"
log "=============================="
python3 -u scripts/zero_shot_diagnostic.py \
    --model Qwen/Qwen2.5-7B \
    --n_problems 100 --k 8 \
    --output results/g1_diagnostic.json \
    2>&1 | tee -a logs/g1_diagnostic.log
log "G1 COMPLETE — check results/g1_diagnostic.json for GO/NO-GO verdict"
log "Continuing regardless (GO assumed); kill queue if NO-GO and revise prefix."

# ── G2: GRPO Qwen2.5-7B baseline ─────────────────────────────────────────────
run_job g2_grpo_qwen25_7b \
    "${COMMON[@]}" \
    --model Qwen/Qwen2.5-7B \
    --lambda_div 0.0 \
    --alpha_diversity 0.0 \
    --output_dir results/rl_runs/g2_grpo_qwen25_7b

log "G2 eval window: run eval + F1/F2/F4/F8 analyses now before G3 starts"
log "Sleep 300s to allow eval to start (adjust as needed)"
sleep 300

# ── G3: COLD-RL (i,k) Qwen2.5-7B seed 1 ─────────────────────────────────────
# PRIMARY RUN — most important result in the paper
run_job g3_cold_rl_qwen25_7b_s1 \
    "${COMMON[@]}" \
    --model Qwen/Qwen2.5-7B \
    --seed 42 \
    --output_dir results/rl_runs/g3_cold_rl_qwen25_7b_s1

# ── G4: COLD-RL (i,k) Qwen2.5-7B seed 2 ─────────────────────────────────────
run_job g4_cold_rl_qwen25_7b_s2 \
    "${COMMON[@]}" \
    --model Qwen/Qwen2.5-7B \
    --seed 1337 \
    --output_dir results/rl_runs/g4_cold_rl_qwen25_7b_s2

# ── G5: COLD-RL (i,k) Qwen2.5-7B seed 3 ─────────────────────────────────────
run_job g5_cold_rl_qwen25_7b_s3 \
    "${COMMON[@]}" \
    --model Qwen/Qwen2.5-7B \
    --seed 0 \
    --output_dir results/rl_runs/g5_cold_rl_qwen25_7b_s3

log "G3/G4/G5 complete — 3 seeds for COLD-RL Qwen2.5-7B done"
log "G6 (ModC reproduction) SKIPPED — requires separate SFT script"
log "  To run G6: python scripts/run_modc.py (not yet implemented)"

# ── G7: COLD-RL Qwen2.5-Math-7B (Power Sampling comparison) ──────────────────
run_job g7_cold_rl_math_7b \
    --dataset math \
    --benchmark math \
    --n_rollouts 8 \
    --temperature 1.0 \
    --total_steps 8000 \
    --n_problems_per_step 32 \
    --mini_batch 8 \
    --ref_batch_size 4 \
    --max_new_tokens 4096 \
    --lr 2e-6 \
    --kl_beta 0.04 \
    --embed_model local \
    --lora_r 32 \
    --lora_alpha 64 \
    --log_every 10 \
    --save_every 1000 \
    --phase1_steps 4000 \
    --alpha_diversity 0.3 \
    --k_max_training 16 \
    --gpu_id 0 \
    --model Qwen/Qwen2.5-Math-7B \
    --seed 42 \
    --output_dir results/rl_runs/g7_cold_rl_math_7b

# ── G8: GRPO 12k steps (compute-control) ─────────────────────────────────────
run_job g8_grpo_12k_qwen25_7b \
    "${COMMON[@]}" \
    --model Qwen/Qwen2.5-7B \
    --lambda_div 0.0 \
    --alpha_diversity 0.0 \
    --total_steps 12000 \
    --output_dir results/rl_runs/g8_grpo_12k_qwen25_7b

# ── G9: k-schedule sparse ablation ({1,4,16}) ────────────────────────────────
run_job g9_sparse_k_qwen25_7b \
    "${COMMON[@]}" \
    --model Qwen/Qwen2.5-7B \
    --k_values 1,4,16 \
    --output_dir results/rl_runs/g9_sparse_k_qwen25_7b

# ── G10: GRPO baseline Qwen2.5-Math-7B (G7 reference) ────────────────────────
run_job g10_grpo_math_7b \
    --dataset math \
    --benchmark math \
    --n_rollouts 8 \
    --temperature 1.0 \
    --total_steps 8000 \
    --n_problems_per_step 32 \
    --mini_batch 8 \
    --ref_batch_size 4 \
    --max_new_tokens 4096 \
    --lr 2e-6 \
    --kl_beta 0.04 \
    --embed_model local \
    --lora_r 32 \
    --lora_alpha 64 \
    --log_every 10 \
    --save_every 1000 \
    --phase1_steps 4000 \
    --alpha_diversity 0.0 \
    --lambda_div 0.0 \
    --k_max_training 16 \
    --gpu_id 0 \
    --model Qwen/Qwen2.5-Math-7B \
    --lambda_div 0.0 \
    --alpha_diversity 0.0 \
    --seed 42 \
    --output_dir results/rl_runs/g10_grpo_math_7b

log "=============================="
log "GH200 QUEUE COMPLETE"
log "All results in results/rl_runs/"
log "=============================="
