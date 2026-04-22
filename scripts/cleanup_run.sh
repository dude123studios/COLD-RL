#!/bin/bash
# Wait for Group M to finish, then re-run to pick up failed experiments
# Also picks up pas_me_k2_2k_3b_v2 from Group L
LOG_M=/home/atharv/hugsim/logs/group_M.log
M_PID=$(pgrep -f "groups M" | head -1)

echo "[cleanup] Waiting for Group M (PID $M_PID) to finish..."
if [ -n "$M_PID" ]; then
    wait $M_PID 2>/dev/null
fi

echo "[cleanup] Group M done. Re-running L+M to catch failures..."
cd /home/atharv/hugsim
CUDA_VISIBLE_DEVICES=2 python -m orchestration.run_experiment --groups L M >> logs/group_LM_cleanup.log 2>&1
echo "[cleanup] Done."
