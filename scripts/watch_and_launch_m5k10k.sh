#!/bin/bash
# Watch for sft_teacher_5k and sft_teacher_10k to complete, then launch PAS-ME variants

LOGDIR="/home/atharv/hugsim/logs"
RESULTS="/home/atharv/hugsim/results/results.json"

echo "[watcher] Waiting for sft_teacher_5k_3b and sft_teacher_10k_3b to complete..."

while true; do
    DONE_5K=$(grep -c '"experiment_id": "sft_teacher_5k_3b"' "$RESULTS" 2>/dev/null; true)
    DONE_5K=${DONE_5K:-0}
    DONE_10K=$(grep -c '"experiment_id": "sft_teacher_10k_3b"' "$RESULTS" 2>/dev/null; true)
    DONE_10K=${DONE_10K:-0}

    # Launch pas_me_k3_5k_3b when sft_teacher_5k_3b completes
    if [ "$DONE_5K" -gt 0 ] && ! pgrep -f "pas_me_k3_5k_3b" > /dev/null 2>&1; then
        DONE_PAS5K=$(grep -c '"experiment_id": "pas_me_k3_5k_3b"' "$RESULTS" 2>/dev/null; true)
        DONE_PAS5K=${DONE_PAS5K:-0}
        if [ "$DONE_PAS5K" -eq 0 ]; then
            echo "[watcher] sft_teacher_5k_3b done, launching pas_me_k3_5k_3b on GPU 3"
            CUDA_VISIBLE_DEVICES=3 nohup python -m orchestration.run_experiment --id pas_me_k3_5k_3b \
                > "$LOGDIR/pas_me_k3_5k.log" 2>&1 &
            echo "[watcher] Launched pas_me_k3_5k_3b (PID: $!)"
        fi
    fi
    
    # Launch pas_me_k3_10k_3b when sft_teacher_10k_3b completes
    if [ "$DONE_10K" -gt 0 ] && ! pgrep -f "pas_me_k3_10k_3b" > /dev/null 2>&1; then
        DONE_PAS10K=$(grep -c '"experiment_id": "pas_me_k3_10k_3b"' "$RESULTS" 2>/dev/null; true)
        DONE_PAS10K=${DONE_PAS10K:-0}
        if [ "$DONE_PAS10K" -eq 0 ]; then
            echo "[watcher] sft_teacher_10k_3b done, launching pas_me_k3_10k_3b on GPU 4"
            CUDA_VISIBLE_DEVICES=4 nohup python -m orchestration.run_experiment --id pas_me_k3_10k_3b \
                > "$LOGDIR/pas_me_k3_10k.log" 2>&1 &
            echo "[watcher] Launched pas_me_k3_10k_3b (PID: $!)"
        fi
    fi
    
    # Exit when both launched
    DONE_PAS5K=$(grep -c '"experiment_id": "pas_me_k3_5k_3b"' "$RESULTS" 2>/dev/null; true)
    DONE_PAS5K=${DONE_PAS5K:-0}
    DONE_PAS10K=$(grep -c '"experiment_id": "pas_me_k3_10k_3b"' "$RESULTS" 2>/dev/null; true)
    DONE_PAS10K=${DONE_PAS10K:-0}
    if [ "$DONE_PAS5K" -gt 0 ] && [ "$DONE_PAS10K" -gt 0 ]; then
        echo "[watcher] Both PAS-ME 5k and 10k complete. Exiting."
        exit 0
    fi
    
    sleep 60
done
