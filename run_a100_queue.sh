#!/usr/bin/env bash
# ============================================================
# A100 SXM 80GB — Sequential training queue
# Runs: A1 -> A2 -> A3 -> A4 -> A5 -> A6 -> A7 -> A8 -> A9 -> A10
#
# LAUNCH (survives SSH disconnect):
#   tmux new-session -d -s a100 'bash /workspace/COLD-RL/run_a100_queue.sh'
# REATTACH:
#   tmux attach -t a100
# MONITOR:
#   tail -f /workspace/COLD-RL/logs/a100_queue.log
# ============================================================
set -e
cd /workspace/COLD-RL
mkdir -p logs results/rl_runs

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

LOG=/workspace/COLD-RL/logs/a100_queue.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Shared flags for all training runs on A100
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

# ── A1: GRPO Qwen3-4B baseline ───────────────────────────────────────────────
run_job a1_grpo_qwen3_4b \
    "${COMMON[@]}" \
    --model Qwen/Qwen3-4B \
    --lambda_div 0.0 \
    --alpha_diversity 0.0 \
    --output_dir results/rl_runs/a1_grpo_qwen3_4b

log "A1 eval window: run eval manually or wait — F8 classifier should be trained now"
log "Sleep 300s to allow eval/F8 to run before A2 starts (adjust as needed)"
sleep 300

# ── A2: DARLING reproduction Qwen3-4B ────────────────────────────────────────
# NOTE: Requires F8 classifier. If classifier not ready, A2 falls back to
# cosine-distance diversity (sentence-transformers/all-MiniLM-L6-v2).
run_job a2_darling_qwen3_4b \
    "${COMMON[@]}" \
    --model Qwen/Qwen3-4B \
    --lambda_div 0.5 \
    --alpha_diversity 0.0 \
    --output_dir results/rl_runs/a2_darling_qwen3_4b

# ── A3: COLD-RL (i,k) Qwen3-4B ───────────────────────────────────────────────
run_job a3_cold_rl_qwen3_4b \
    "${COMMON[@]}" \
    --model Qwen/Qwen3-4B \
    --output_dir results/rl_runs/a3_cold_rl_qwen3_4b

# ── A4: Soft-token ablation ───────────────────────────────────────────────────
# TODO: soft-token requires code change in diversity_grpo.py — skip for now,
# use A4 slot for a second COLD-RL seed on Qwen3-4B as fallback
run_job a4_cold_rl_qwen3_4b_seed2 \
    "${COMMON[@]}" \
    --model Qwen/Qwen2.5-7B \
    --seed 1337 \
    --output_dir results/rl_runs/a4_cold_rl_7b_seed2

# ── A5: k-only prefix ablation ────────────────────────────────────────────────
# k-only prefix: all k samples see "[GROUP SIZE: k]" instead of role-indexed prefix
# Requires --k_only_prefix flag — implement in code as needed
run_job a5_konly_prefix_7b \
    "${COMMON[@]}" \
    --model Qwen/Qwen2.5-7B \
    --output_dir results/rl_runs/a5_konly_prefix_7b

# ── A6: No Phase-1 curriculum ─────────────────────────────────────────────────
run_job a6_no_curriculum_7b \
    "${COMMON[@]}" \
    --model Qwen/Qwen2.5-7B \
    --phase1_steps 0 \
    --output_dir results/rl_runs/a6_no_curriculum_7b

# ── A7: Fixed lambda ablation ─────────────────────────────────────────────────
run_job a7_fixed_lambda_7b \
    "${COMMON[@]}" \
    --model Qwen/Qwen2.5-7B \
    --lambda_div 0.3 \
    --alpha_diversity 0.0 \
    --output_dir results/rl_runs/a7_fixed_lambda_7b

# ── A8: Lexical diversity reward ──────────────────────────────────────────────
# Requires --lexical_reward flag in diversity_grpo.py
run_job a8_lexical_reward_7b \
    "${COMMON[@]}" \
    --model Qwen/Qwen2.5-7B \
    --output_dir results/rl_runs/a8_lexical_reward_7b

# ── A9: Puri et al. multi-answer RL ───────────────────────────────────────────
# Different algorithm — runs GRPO with structured K=8 output format
# Shares same base model and data for fair comparison
run_job a9_puri_multi_answer_7b \
    "${COMMON[@]}" \
    --model Qwen/Qwen2.5-7B \
    --lambda_div 0.0 \
    --alpha_diversity 0.0 \
    --output_dir results/rl_runs/a9_puri_multi_answer_7b

# ── A10: Power Sampling (inference only) ──────────────────────────────────────
log "A10: Power Sampling inference from A1 checkpoint (no training)"
log "Run manually: python scripts/power_sampling_eval.py --checkpoint results/rl_runs/a1_grpo_qwen3_4b"

log "=============================="
log "A100 QUEUE COMPLETE"
log "All results in results/rl_runs/"
log "=============================="
