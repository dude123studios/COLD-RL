#!/bin/bash
# Wait for GPU 2 to have >30GB free memory AND no competing experiment on GPU 2,
# then launch k=5. Re-checks every 60s.

LOGDIR="/home/atharv/hugsim/logs"
RESULTS="/home/atharv/hugsim/results/results.json"

echo "[watcher_k5] Monitoring GPU 2 for k=5 launch opportunity..."

while true; do
    # Check if k=5 already complete
    DONE=$(grep -c '"experiment_id": "pas_me_k5_2k_3b_v2"' "$RESULTS" 2>/dev/null; true)
    DONE=${DONE:-0}
    if [ "$DONE" -gt 0 ]; then
        echo "[watcher_k5] pas_me_k5_2k_3b_v2 already complete. Exiting."
        exit 0
    fi

    # Check GPU 2 free memory
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2 2>/dev/null | tr -d ' ')
    FREE=${FREE:-0}

    if [ "$FREE" -gt 30000 ]; then
        # Check no run_experiment is on GPU 2 right now
        # (by checking if any run_experiment process has CUDA_VISIBLE_DEVICES=2)
        GPU2_BUSY=$(for pid in $(pgrep -f "run_experiment"); do
            env_val=$(cat /proc/$pid/environ 2>/dev/null | tr '\0' '\n' | grep "^CUDA_VISIBLE_DEVICES=2$")
            if [ -n "$env_val" ]; then echo "busy"; fi
        done)

        if [ -z "$GPU2_BUSY" ]; then
            echo "[watcher_k5] GPU 2 has ${FREE}MB free and is idle. Launching pas_me_k5_2k_3b_v2..."
            CUDA_VISIBLE_DEVICES=2 nohup python -m orchestration.run_experiment --id pas_me_k5_2k_3b_v2 \
                > "$LOGDIR/pas_me_k5.log" 2>&1 &
            echo "[watcher_k5] Launched PID: $!"
            exit 0
        else
            echo "[watcher_k5] GPU 2 has ${FREE}MB free but another run_experiment is active. Waiting..."
        fi
    else
        echo "[watcher_k5] GPU 2 has ${FREE}MB free, need >30000MB. Waiting..."
    fi

    sleep 60
done
