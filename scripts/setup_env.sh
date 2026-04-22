#!/bin/bash
# One-time setup: create venv and install dependencies.
# Run once before launching experiments.
#
# Usage: ./scripts/setup_env.sh

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[setup] Creating virtualenv..."
python3 -m venv "$REPO_DIR/.venv"
source "$REPO_DIR/.venv/bin/activate"

echo "[setup] Installing dependencies..."
pip install --upgrade pip
pip install -r "$REPO_DIR/requirements.txt"

echo "[setup] Done. Activate with: source $REPO_DIR/.venv/bin/activate"
echo "[setup] Set your API key: export OPENROUTER_API_KEY=<your-key>"
