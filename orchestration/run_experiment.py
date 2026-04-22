"""
Main experiment orchestrator.

Reads experiment_registry.yaml, finds incomplete experiments,
and runs them in group/priority order. Each experiment is atomic:
  build dataset → train → eval all benchmarks → save results → delete LoRA

Crash-safe: results.json is append-only. Rerunning skips completed experiments.
"""

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import yaml

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_FILE = RESULTS_DIR / "results.json"
REGISTRY_FILE = Path(__file__).parent / "experiment_registry.yaml"
ADAPTER_DIR = RESULTS_DIR / "adapters"

GROUP_ORDER = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
               "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
               "BASELINE", "CFT", "EDF"]


def load_registry() -> list[dict]:
    with open(REGISTRY_FILE) as f:
        return yaml.safe_load(f)["experiments"]


def load_completed() -> set[str]:
    if not RESULTS_FILE.exists():
        return set()
    completed = set()
    with open(RESULTS_FILE) as f:
        for line in f:
            try:
                row = json.loads(line)
                completed.add(row["experiment_id"])
            except Exception:
                pass
    return completed


def save_result(experiment_id: str, result: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "experiment_id": experiment_id,
        "timestamp": datetime.utcnow().isoformat(),
        **result,
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[results] Saved {experiment_id}")


def check_dependencies(config: dict, completed: set[str]) -> bool:
    dep = config.get("depends_on")
    if dep and dep not in completed:
        print(f"[skip] {config['id']} waiting for dependency: {dep}")
        return False
    return True


def resolve_lora_path(experiment_id: str) -> str:
    return str(ADAPTER_DIR / experiment_id)


def _free_gpu_memory() -> None:
    """Force-release GPU memory held by training before launching vLLM."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            print(f"[gpu] Free after cleanup: {free/1e9:.1f}/{total/1e9:.1f} GiB")
    except Exception:
        pass


def _reset_gpu_state() -> None:
    """Hard-reset CUDA context between experiments to prevent error propagation."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            # Destroy and reinitialize the CUDA context on all devices
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
            # Force Python to release all GPU tensor references
            gc.collect()
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info(0)
            print(f"[gpu] Reset complete. Free: {free/1e9:.1f}/{total/1e9:.1f} GiB", flush=True)
    except Exception as e:
        print(f"[gpu] Reset warning: {e}", flush=True)


def _build_rare_dataset(config: dict) -> tuple:
    """
    For Group H experiments: select rare problem, generate family, build dataset.
    Returns (dataset_path, rare_problem_dict, family_list).
    """
    import json as _json
    from pathlib import Path as _Path
    from generation.build_sft_dataset import load_math_train, save_dataset
    from generation.rare_problem import (
        select_rare_problem,
        generate_problem_family,
        build_rare_experiment_dataset,
    )
    from evaluation.vllm_generate import eval_generate as _eval_gen, load_benchmark

    DATASET_DIR = RESULTS_DIR / "datasets"
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    exp_id = config["id"]
    training_cfg = config.get("training", {})
    strategy = training_cfg["strategy"]
    k = training_cfg.get("k", 3)
    family_size = training_cfg.get("family_size", 5)
    model = config["model"]

    dataset_path = str(DATASET_DIR / f"{exp_id}.jsonl")
    rare_meta_path = RESULTS_DIR / "rare_problem_meta.json"

    # Load or select rare problem
    if rare_meta_path.exists():
        with open(rare_meta_path) as f:
            rare_problem = _json.load(f)
        print(f"[rare] Using cached rare problem: {rare_problem['id']}")
    else:
        # Need base model rollouts to select rare problem
        # These come from baseline_3b eval cache
        problems = load_math_train()
        rollouts = _eval_gen(
            base_model=model,
            lora_path=None,
            benchmark="math500",
            n=16,
            temperature=0.6,
        )
        # For training problem selection, also eval training problems (use cached rollouts)
        rare_problem = select_rare_problem(problems, rollouts)
        with open(rare_meta_path, "w") as f:
            _json.dump(rare_problem, f, indent=2)

    # Generate problem family (cached inside generate_problem_family)
    rare_family = generate_problem_family(
        rare_problem,
        m=family_size,
        model=training_cfg.get("teacher", "deepseek/deepseek-r1-distill-qwen-32b"),
    )

    if _Path(dataset_path).exists():
        print(f"[dataset] Using cached rare dataset: {dataset_path}")
        return dataset_path, rare_problem, rare_family

    base_problems = load_math_train(n_problems=training_cfg.get("dataset_size"))
    rows = build_rare_experiment_dataset(base_problems, rare_problem, strategy, k=k)
    save_dataset(rows, dataset_path)
    return dataset_path, rare_problem, rare_family


def run_experiment(config: dict, keep_adapter: bool = False) -> dict:
    """Run one experiment end-to-end. Returns results dict."""
    from generation.build_sft_dataset import build_dataset_for_config
    from training.lora_lifecycle import LoRALifecycle, get_checkpoint_paths_with_pct
    from evaluation.vllm_generate import eval_generate, score_answers, load_benchmark
    from evaluation.compute_rdiv import compute_rdiv_dataset

    exp_id = config["id"]
    model = config["model"]
    training_cfg = config.get("training", {})
    strategy = training_cfg.get("strategy", "none")
    benchmarks = config.get("eval_benchmarks", ["math500"])
    n_eval = config.get("n_eval", 16)
    temperature = config.get("temperature", 0.6)
    checkpoint_pcts = config.get("checkpoint_evals", [])

    results: dict = {"model": model, "training_config": training_cfg}

    # ── Step 1: Build dataset (offline, before GPU work) ──────────────────
    adapter_path = None
    rare_problem = None
    rare_family = None

    # eval_only: skip all training, load an existing adapter by experiment_id
    adapter_from = config.get("adapter_from")
    if adapter_from:
        resolved = resolve_lora_path(adapter_from)
        if not Path(resolved).exists():
            raise RuntimeError(
                f"adapter_from={adapter_from!r} not found at {resolved}. "
                f"Run that experiment first."
            )
        adapter_path = resolved
        strategy = "none"  # skip training branch entirely

    if strategy in ("sft_rare", "multi_sft_rare", "pas_rare"):
        config["dataset_path"], rare_problem, rare_family = _build_rare_dataset(config)
    elif strategy in ("sdft", "sdft_pas"):
        from generation.sdft_collect import collect_sdft_data
        from generation.build_sft_dataset import load_math_train
        base_problems = load_math_train(n_problems=training_cfg.get("dataset_size"))

        if strategy == "sdft_pas":
            # Generate PAS alternatives first, use each as teacher context.
            # Each alternative becomes a separate training example with a unique id.
            from generation.pas_augment import PASAugmenter
            augmenter = PASAugmenter(
                teacher_model=training_cfg.get(
                    "teacher", "deepseek/deepseek-r1-distill-qwen-32b"
                ),
            )
            pas_rows = augmenter.augment_dataset(
                base_problems,
                k=training_cfg.get("k", 3),
                context_mode=training_cfg.get("context_mode", "full"),
                experiment_id=f"sdftpas_{exp_id}",
                include_original=False,
            )
            # Give each PAS alternative a unique id so collect_sdft_data caches them separately
            train_problems = [
                {**row, "id": f"{row['id']}_{row['source']}"}
                for row in pas_rows
            ]
        else:
            train_problems = base_problems

        config["sdft_cache_dir"] = collect_sdft_data(
            model_name=model,
            problems=train_problems,
            max_new_tokens=training_cfg.get("max_new_tokens", 8192),
            top_k=training_cfg.get("top_k", 50),
            temperature=training_cfg.get("collect_temperature", 0.6),
            experiment_id=exp_id,
        )
    elif strategy != "none":
        config["dataset_path"] = build_dataset_for_config(config)

    # ── Step 2: Train + Eval (inside lifecycle for cleanup) ───────────────
    with LoRALifecycle(exp_id, keep_adapter=keep_adapter) as lifecycle:

        if strategy not in ("none",):
            if strategy in ("sdft", "sdft_pas"):
                from training.sdft_train import train_sdft
                adapter_path = train_sdft(config, exp_id)
            elif strategy == "grpo":
                from training.grpo_train import train_grpo

                # Resolve init_from_lora reference to actual path
                if training_cfg.get("init_from_lora"):
                    ref_id = training_cfg["init_from_lora"]
                    ref_path = resolve_lora_path(ref_id)
                    if not Path(ref_path).exists():
                        raise RuntimeError(
                            f"Dependency adapter not found: {ref_path}. "
                            f"Run {ref_id} first."
                        )
                    config["training"]["init_from_lora"] = ref_path

                adapter_path = train_grpo(config, exp_id)
            else:
                from training.sft_train import train_sft
                adapter_path = train_sft(config, exp_id)

            lifecycle.register(adapter_path)

        # Free GPU memory held by training before vLLM tries to allocate
        _free_gpu_memory()

        # ── Step 3: Eval on all benchmarks ────────────────────────────────
        for benchmark in benchmarks:
            print(f"[eval] {exp_id} — {benchmark}")
            problems = load_benchmark(benchmark)

            rollouts = eval_generate(
                base_model=model,
                lora_path=adapter_path,
                benchmark=benchmark,
                n=n_eval,
                temperature=temperature,
            )

            scores = score_answers(rollouts, problems, benchmark=benchmark)
            rdiv = compute_rdiv_dataset(rollouts, problems) if benchmark == "math500" else {}

            results[benchmark] = {**scores, **rdiv}
            print(f"[eval] {benchmark}: {scores}")

        # ── Step 3b: Rare problem family eval (Group H only) ─────────────
        if config.get("eval_rare_family") and rare_problem and rare_family:
            # Eval on the rare problem itself
            rare_as_benchmark = [rare_problem] + rare_family
            rollouts_rare = eval_generate(
                base_model=model,
                lora_path=adapter_path,
                benchmark="math500",
                n=n_eval,
                temperature=temperature,
                extra_problems=rare_as_benchmark,
            )
            # Score only on rare problem and its variants
            rare_ids = {p["id"] for p in rare_as_benchmark}
            rollouts_rare_only = {k: v for k, v in rollouts_rare.items() if k in rare_ids}
            scores_rare = score_answers(rollouts_rare_only, rare_as_benchmark)

            # Separate rare problem score from family scores
            rare_rolls = {rare_problem["id"]: rollouts_rare.get(rare_problem["id"], [])}
            family_rolls = {p["id"]: rollouts_rare.get(p["id"], []) for p in rare_family}

            score_rare_only = score_answers(rare_rolls, [rare_problem])
            score_family = score_answers(family_rolls, rare_family)

            gen_gap = {
                k.replace("pass_at", "gap_at"): (score_rare_only.get(k, 0) - score_family.get(k, 0))
                for k in score_rare_only
            }

            results["rare_problem"] = {
                **score_rare_only,
                "problem_id": rare_problem["id"],
            }
            results["rare_family"] = {
                **score_family,
                "n_variants": len(rare_family),
            }
            results["generalization_gap"] = gen_gap
            print(f"[eval] rare_problem: {score_rare_only}")
            print(f"[eval] rare_family:  {score_family}")
            print(f"[eval] gen_gap:      {gen_gap}")

        # ── Step 4: Checkpoint evals ───────────────────────────────────────
        if adapter_path and checkpoint_pcts:
            from training.sft_train import ADAPTER_DIR as _ADIR
            import math

            # Estimate total steps from training config
            ds_size_approx = 12500 * (1 + training_cfg.get("k", 0))
            steps = math.ceil(
                ds_size_approx
                / (training_cfg.get("batch_size", 2) * training_cfg.get("grad_accum", 8))
            ) * training_cfg.get("epochs", 1)

            ckpt_pairs = get_checkpoint_paths_with_pct(adapter_path, steps)
            for ckpt_path, pct in ckpt_pairs:
                lifecycle.register(ckpt_path)
                if pct not in checkpoint_pcts:
                    continue
                print(f"[eval] checkpoint {pct}% — math500")
                problems = load_benchmark("math500")
                rollouts = eval_generate(
                    base_model=model,
                    lora_path=ckpt_path,
                    benchmark="math500",
                    n=n_eval,
                    temperature=temperature,
                )
                scores = score_answers(rollouts, problems)
                rdiv = compute_rdiv_dataset(rollouts, problems)
                results[f"ckpt_{pct}"] = {**scores, **rdiv}

    # LoRA adapter deleted here by lifecycle __exit__

    return results


def run_all(groups: list[str] = None, only_id: str = None) -> None:
    """
    Run all experiments (or a subset) in group/priority order.

    Args:
        groups: list of group letters to run (e.g. ["A", "B"]). None = all.
        only_id: if set, run only this experiment_id.
    """
    registry = load_registry()
    completed = load_completed()

    if only_id:
        registry = [e for e in registry if e["id"] == only_id]
        if not registry:
            print(f"[error] Experiment {only_id} not found in registry")
            sys.exit(1)

    # Sort by group then priority
    def sort_key(e):
        g = e.get("group", "Z")
        group_idx = GROUP_ORDER.index(g) if g in GROUP_ORDER else 99
        return (group_idx, e.get("priority", 99))

    registry = sorted(registry, key=sort_key)

    # Build set of experiment IDs whose adapters are needed by init_from_lora or adapter_from
    all_registry = load_registry()
    lora_sources = {
        e["training"]["init_from_lora"]
        for e in all_registry
        if e.get("training", {}).get("init_from_lora")
    } | {
        e["adapter_from"]
        for e in all_registry
        if e.get("adapter_from")
    }

    if groups:
        registry = [e for e in registry if e.get("group", "") in groups]

    print(f"[run_all] {len(registry)} experiments in queue")
    print(f"[run_all] Already completed: {len(completed)}")

    for config in registry:
        exp_id = config["id"]

        if exp_id in completed:
            print(f"[skip] {exp_id} — already complete")
            continue

        if not check_dependencies(config, completed):
            continue

        print(f"\n{'='*60}")
        print(f"[start] {exp_id}")
        print(f"{'='*60}")

        _reset_gpu_state()

        try:
            keep_adapter = exp_id in lora_sources
            if keep_adapter:
                print(f"[lifecycle] Keeping adapter for {exp_id} (referenced by init_from_lora)")
            result = run_experiment(config, keep_adapter=keep_adapter)
            save_result(exp_id, result)
            completed.add(exp_id)
        except Exception as e:
            print(f"[error] {exp_id} failed: {e}")
            traceback.print_exc()
            _reset_gpu_state()  # clean up after failure before next experiment
            continue

    print(f"\n[done] Completed {len(completed)} experiments total")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run HugSim experiments")
    parser.add_argument("--groups", nargs="+", help="Groups to run (e.g. A B C)")
    parser.add_argument("--id", help="Run only this experiment_id")
    args = parser.parse_args()

    run_all(groups=args.groups, only_id=args.id)
