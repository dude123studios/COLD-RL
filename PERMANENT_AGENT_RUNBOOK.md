# Hugsim RL runs — agent runbook (permanent)

## Operating policy

**Always fix blockers and keep training running.** If jobs are down, diagnose from logs, patch code or launch args, then relaunch (typically `restart2.sh`). Do not stop after reporting status only.

## What is running

Four **Diversity-GRPO / COLD** jobs on **48 GB** GPUs (when available):

| Job              | Lambda | Output dir                                      | Log                    |
|-----------------|--------|-------------------------------------------------|------------------------|
| cold λ=0.5      | 0.5    | `results/rl_runs/cold_deepscaler_lambda05_7b`   | `logs/rl_opt_lam05.log` |
| cold λ=0.1      | 0.1    | `results/rl_runs/cold_deepscaler_lambda010_7b`  | `logs/rl_opt_lam01.log` |
| cold λ=0.7      | 0.7    | `results/rl_runs/cold_deepscaler_lambda070_7b`  | `logs/rl_opt_lam07.log` |
| GRPO baseline   | 0.0    | `results/rl_runs/grpo_baseline_deepscaler_7b`   | `logs/rl_opt_grpo.log`  |

Resume checkpoints in `restart2.sh` may need updating as training progresses (e.g. `step_00100` → latest `step_*`).

## Relaunch (primary)

From repo root:

```bash
cd /home/atharv/hugsim && bash restart2.sh
```

Script kills stale `run_rl` / `EngineCore`, sets `PYTORCH_ALLOC_CONF=expandable_segments:True`, uses `ref_batch_size 2`, `mini_batch 8`, launches four `nohup` jobs on GPUs 0–3.

## Health checks

```bash
pgrep -af rl.run_rl
nvidia-smi
tail -n 80 logs/rl_opt_lam05.log
tail -n 5 results/rl_runs/cold_deepscaler_lambda05_7b/training_log.jsonl
```

## Common failures and fixes

### vLLM: free memory \< desired utilization

**Symptom:** `ValueError: Free memory on device cuda:0 (...) is less than desired GPU memory utilization (0.45, ...)`.

**Cause:** `gpu_memory_utilization` must satisfy `util * total_vram ≤ driver_free`. Sizing util from `torch.cuda.memory_allocated()` only is unsafe; use **`torch.cuda.mem_get_info()`** (see `create_vllm_engine` in `rl/diversity_grpo.py`).

### vLLM EngineCore: CUDA OOM during worker startup

**Symptom:** `EngineCore failed to start` / `torch.OutOfMemoryError` in the child after HF weights are already on the GPU (often on **second and later** vLLM inits in the same process, when memory is fragmented).

**Mitigations in code:** conservative **util cap** (e.g. 0.38), **larger slack** subtracted from driver free, `empty_cache` + `synchronize` before each init, and **automatic retries** that multiply `gpu_memory_utilization` by ~0.88 until init succeeds or the floor is hit.

### OOM during ref log-probs or backward with vLLM still loaded

**Fix:** Tear down vLLM after rollouts before ref forward/backward (`destroy_vllm_engine`); keep `ref_batch_size` small; chunked logsumexp for large vocab.

### Stale / zombie engines

```bash
pkill -9 -f run_rl
pkill -9 -f EngineCore
sleep 10
nvidia-smi
```

Then `restart2.sh`.

## Key implementation files

- `rl/diversity_grpo.py` — training loop, vLLM create/destroy, ref log-probs
- `rl/run_rl.py` — CLI
- `restart2.sh` — four-job launcher

## GPU layout note

If GPUs 0–3 are busy and 4–7 are free (or the reverse), edit GPU IDs in `restart2.sh` to match the machine. The training code uses `CUDA_VISIBLE_DEVICES` so process-local `cuda:0` is the chosen physical GPU.
