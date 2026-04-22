#!/bin/bash
# Launch all HugSim experiments in a persistent tmux session.
# Experiments survive terminal close. Re-running is safe (skips completed ones).
#
# Usage:
#   ./scripts/launch_tmux.sh              # run all groups
#   ./scripts/launch_tmux.sh A B          # run groups A and B only
#   ./scripts/launch_tmux.sh --id pas_k3_3b_full  # run one experiment
#
# To attach to a running session:
#   tmux attach -t hugsim
#
# To see live logs:
#   tmux attach -t hugsim
#   (or)
#   tail -f logs/hugsim.log

set -e

SESSION="hugsim"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

# Build tmux command from args
if [[ "$1" == "--id" ]]; then
    RUN_CMD="python -m orchestration.run_experiment --id $2"
    LOG_FILE="$LOG_DIR/${2}.log"
elif [[ $# -gt 0 ]]; then
    RUN_GROUPS="$*"
    RUN_CMD="python -m orchestration.run_experiment --groups $RUN_GROUPS"
    LOG_FILE="$LOG_DIR/groups_${RUN_GROUPS// /_}.log"
else
    RUN_CMD="python -m orchestration.run_experiment"
    LOG_FILE="$LOG_DIR/hugsim_$(date +%Y%m%d_%H%M%S).log"
fi

# Kill existing session if present
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create new detached session
tmux new-session -d -s "$SESSION" -x 220 -y 50

# Set working directory, activate env, then run
tmux send-keys -t "$SESSION" "cd $REPO_DIR && export PYTHONPATH=$REPO_DIR:\$PYTHONPATH && source $REPO_DIR/scripts/activate_env.sh; $RUN_CMD 2>&1 | tee $LOG_FILE" Enter

echo "========================================"
echo "  HugSim experiments launched in tmux"
echo "  Session: $SESSION"
echo "  Log:     $LOG_FILE"
echo "========================================"
echo ""
echo "Attach:  tmux attach -t $SESSION"
echo "Monitor: tail -f $LOG_FILE"
echo "Kill:    tmux kill-session -t $SESSION"
