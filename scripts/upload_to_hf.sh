#!/usr/bin/env bash
# Upload the final COLD-RL checkpoint to Hugging Face Hub.
# Run ONLY after training completes (all steps done).
# Never called automatically during training — step checkpoints stay local.
#
# Usage:
#   HF_TOKEN=hf_... bash scripts/upload_to_hf.sh \
#       /data/user_data/shivansg/cold_rl_runs/cold_rl_main \
#       dude123studios/cold-rl-qwen25-7b

set -euo pipefail

CHECKPOINT_DIR="${1:-/data/user_data/shivansg/cold_rl_runs/cold_rl_main}"
HF_REPO="${2:-dude123studios/cold-rl-final}"
HF_TOKEN="${HF_TOKEN:-}"

if [[ -z "${HF_TOKEN}" ]]; then
    echo "Error: HF_TOKEN env var is not set."
    echo "Usage: HF_TOKEN=hf_... bash scripts/upload_to_hf.sh <checkpoint_dir> <hf_repo>"
    exit 1
fi

# Find the highest-numbered step checkpoint
FINAL_CKPT=$(
    find "${CHECKPOINT_DIR}" -maxdepth 1 -type d -name "step_*" \
    | sort -V \
    | tail -1
)

if [[ -z "${FINAL_CKPT}" ]]; then
    echo "Error: No step_XXXXX checkpoint found in ${CHECKPOINT_DIR}"
    exit 1
fi

STEP=$(basename "${FINAL_CKPT}" | sed 's/step_//')
echo "[upload] Final checkpoint : ${FINAL_CKPT}  (step ${STEP})"
echo "[upload] Target HF repo   : ${HF_REPO}"

python3 - <<PYEOF
from huggingface_hub import HfApi, upload_folder

api = HfApi(token="${HF_TOKEN}")
repo_id = "${HF_REPO}"
checkpoint_dir = "${FINAL_CKPT}"

try:
    api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
    print(f"[upload] Repo {repo_id} ready")
except Exception as e:
    print(f"[upload] create_repo warning (may already exist): {e}")

upload_folder(
    folder_path=checkpoint_dir,
    repo_id=repo_id,
    token="${HF_TOKEN}",
    commit_message="Final COLD-RL checkpoint — step ${STEP}",
    ignore_patterns=["*.tmp", "*.lock", "optimizer.pt", "scheduler.pt"],
)
print(f"[upload] Done → https://huggingface.co/{repo_id}")
PYEOF
