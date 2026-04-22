"""
LoRA lifecycle context manager.

Guarantees cleanup of adapter files after eval completes,
even if eval crashes. Prevents disk accumulation across long experiment runs.
"""

import shutil
import os
from pathlib import Path


class LoRALifecycle:
    """
    Context manager that tracks and deletes LoRA adapter directories on exit.

    Usage:
        with LoRALifecycle(experiment_id) as lifecycle:
            adapter_path = train_sft(config)
            lifecycle.register(adapter_path)
            results = eval_all(adapter_path)
            save_results(results)
        # adapter deleted here, even on exception
    """

    def __init__(self, experiment_id: str, keep_checkpoints: bool = False, keep_adapter: bool = False):
        self.experiment_id = experiment_id
        self.keep_checkpoints = keep_checkpoints
        self.keep_adapter = keep_adapter
        self.adapter_paths: list[str] = []

    def register(self, path: str) -> str:
        """Register a path for cleanup. Returns the path for chaining."""
        if path and path not in self.adapter_paths:
            self.adapter_paths.append(path)
        return path

    def cleanup(self) -> None:
        if self.keep_adapter:
            print(f"[lifecycle] Keeping adapter(s) (keep_adapter=True): {self.adapter_paths}")
            self.adapter_paths.clear()
            return
        for path in self.adapter_paths:
            if os.path.exists(path):
                shutil.rmtree(path)
                print(f"[lifecycle] Deleted {path}")
            else:
                print(f"[lifecycle] Already gone: {path}")
        self.adapter_paths.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False  # do not suppress exceptions


def get_checkpoint_paths(adapter_dir: str) -> list[str]:
    """Return sorted list of checkpoint subdirs inside adapter_dir."""
    p = Path(adapter_dir)
    if not p.exists():
        return []
    return sorted(
        str(c) for c in p.iterdir()
        if c.is_dir() and c.name.startswith("checkpoint-")
    )


def get_checkpoint_paths_with_pct(
    adapter_dir: str, total_steps: int
) -> list[tuple[str, int]]:
    """
    Return (path, pct) pairs where pct is approximate training completion.
    Percentages: 25, 50, 75, 100
    """
    checkpoints = get_checkpoint_paths(adapter_dir)
    if not checkpoints or total_steps == 0:
        return []

    result = []
    for ckpt_path in checkpoints:
        step_str = Path(ckpt_path).name.replace("checkpoint-", "")
        try:
            step = int(step_str)
            pct = round(100 * step / total_steps)
            result.append((ckpt_path, pct))
        except ValueError:
            continue
    return result
