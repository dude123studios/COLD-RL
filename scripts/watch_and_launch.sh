#!/bin/bash
# Wait for a GPU with >=40GB free, then launch Group N
TARGET_FREE=40000  # MiB
LOG=/home/atharv/hugsim/logs/group_N.log

echo "[watch] Waiting for GPU with ${TARGET_FREE}MB free..."
while true; do
    while IFS=',' read -r idx free; do
        idx=$(echo $idx | tr -d ' ')
        free=$(echo $free | tr -d ' MiB')
        if [ "$free" -gt "$TARGET_FREE" ] 2>/dev/null; then
            echo "[watch] GPU $idx has ${free}MiB free — launching Group N"
            CUDA_VISIBLE_DEVICES=$idx nohup python -m orchestration.run_experiment --groups N > $LOG 2>&1 &
            echo "[watch] Group N PID: $!"
            exit 0
        fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader)
    sleep 60
done
