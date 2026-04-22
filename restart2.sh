#!/usr/bin/env bash
# Restart all 4 training runs with fixed persistent vLLM + batched ref log-probs.
# Changes from previous attempt:
#   - vLLM util cap: 0.45 (was 0.48) -> more headroom for activations
#   - ref_batch_size: 2 (safe lm_head logits on 48GB after vLLM teardown)
#   - memory-efficient log-prob: logsumexp trick (no full vocab log_softmax tensor)
#   - PYTORCH_ALLOC_CONF=expandable_segments:True -> reduce fragmentation
#   - mini_batch: 8 (was 16) -> halves backward activation peak
set -e
cd "$(dirname "$0")"

echo "[$(date)] Killing old run_rl processes..."
pkill -9 -f "run_rl" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
sleep 10

echo "[$(date)] GPU state after kill:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

mkdir -p logs

COMMON_ARGS=(
    --model Qwen/Qwen2.5-7B-Instruct
    --dataset deepscaler
    --benchmark math
    --n_rollouts 8
    --total_steps 1000
    --n_problems_per_step 32
    --mini_batch 8
    --ref_batch_size 2
    --max_new_tokens 4096
    --lr 1e-6
    --kl_beta 0.001
    --embed_model local
    --lora_r 64
    --log_every 10
    --save_every 100
)

run_job() {
    local GPU="$1"; local LAM="$2"; local OUTDIR="$3"; local RESUME="$4"; local LOG="$5"
    echo "[$(date)] Launching lambda=$LAM on GPU $GPU -> $LOG"
    CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_ALLOC_CONF=expandable_segments:True \
        nohup python3 -u -m rl.run_rl \
            "${COMMON_ARGS[@]}" \
            --lambda_div "$LAM" \
            --output_dir "$OUTDIR" \
            --gpu_id "$GPU" \
            --resume_from "$RESUME" \
        > "$LOG" 2>&1 &
    echo "  PID=$!"
}

run_job 0 0.5 results/rl_runs/cold_deepscaler_lambda05_7b  results/rl_runs/cold_deepscaler_lambda05_7b/step_00100  logs/rl_opt_lam05.log
run_job 1 0.1 results/rl_runs/cold_deepscaler_lambda010_7b results/rl_runs/cold_deepscaler_lambda010_7b/step_00100 logs/rl_opt_lam01.log
run_job 2 0.7 results/rl_runs/cold_deepscaler_lambda070_7b results/rl_runs/cold_deepscaler_lambda070_7b/step_00100 logs/rl_opt_lam07.log
# Use GPU 3 when 4–7 are occupied by other jobs (adjust if a 48GB card frees up).
run_job 3 0.0 results/rl_runs/grpo_baseline_deepscaler_7b  results/rl_runs/grpo_baseline_deepscaler_7b/step_00400  logs/rl_opt_grpo.log

echo "[$(date)] All 4 jobs launched. Monitor with:"
echo "  tail -f logs/rl_opt_lam05.log"
echo "  watch -n30 nvidia-smi"
