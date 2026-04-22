#!/bin/bash
# Activate the Python environment for HugSim.
# Edit VENV_PATH to match your setup.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Try conda first, then venv
if command -v conda &>/dev/null && conda env list | grep -q "hugsim"; then
    conda activate hugsim
elif [[ -f "$REPO_DIR/.venv/bin/activate" ]]; then
    source "$REPO_DIR/.venv/bin/activate"
elif [[ -n "$VIRTUAL_ENV" ]]; then
    echo "[env] Already in virtualenv: $VIRTUAL_ENV"
else
    echo "[env] WARNING: No environment found. Install deps with:"
    echo "  pip install -r $REPO_DIR/requirements.txt"
fi

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"

echo "[env] PYTHONPATH=$REPO_DIR"
