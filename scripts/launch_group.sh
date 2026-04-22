#!/bin/bash
# Launch a specific group in a named tmux window within the hugsim session.
# Useful for running Group A in one window while monitoring in another.
#
# Usage: ./scripts/launch_group.sh A

set -e

GROUP="${1:-A}"
SESSION="hugsim"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/group_${GROUP}_$(date +%Y%m%d_%H%M%S).log"

# Create session if it doesn't exist
tmux new-session -d -s "$SESSION" -x 220 -y 50 2>/dev/null || true

# Create a new window for this group
tmux new-window -t "$SESSION" -n "group_$GROUP"
tmux send-keys -t "$SESSION:group_$GROUP" "cd $REPO_DIR" Enter
tmux send-keys -t "$SESSION:group_$GROUP" "source $REPO_DIR/scripts/activate_env.sh 2>/dev/null || true" Enter
tmux send-keys -t "$SESSION:group_$GROUP" \
    "python -m orchestration.run_experiment --groups $GROUP 2>&1 | tee $LOG_FILE" Enter

echo "Group $GROUP launched in tmux window 'group_$GROUP'"
echo "Attach: tmux attach -t $SESSION"
