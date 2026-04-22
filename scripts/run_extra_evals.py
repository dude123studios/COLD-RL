"""
Run GPQA diamond + livecodebench evals on already-completed experiments
that only have math500 scores. Updates results.json in-place.

Usage:
  python -m scripts.run_extra_evals --groups Z
  python -m scripts.run_extra_evals --id baseline_llama3b
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

RESULTS_FILE = Path("results/results.json")
ADAPTERS_DIR = Path("results/adapters")


def load_results() -> list[dict]:
    if not RESULTS_FILE.exists():
        return []
    return [json.loads(l) for l in RESULTS_FILE.read_text().splitlines() if l.strip()]


def save_results(rows: list[dict]):
    RESULTS_FILE.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def run_extra_evals(
    extra_benchmarks: list[str],
    groups: list[str] = None,
    only_id: str = None,
    n_eval: int = 16,
    temperature: float = 0.6,
):
    import yaml
    from evaluation.vllm_generate import eval_generate, score_answers, load_benchmark
    from evaluation.compute_rdiv import compute_rdiv_dataset

    registry = yaml.safe_load(open("orchestration/experiment_registry.yaml"))["experiments"]
    config_by_id = {e["id"]: e for e in registry}

    rows = load_results()
    updated = False

    for i, row in enumerate(rows):
        eid = row["experiment_id"]

        if only_id and eid != only_id:
            continue

        config = config_by_id.get(eid)
        if not config:
            continue

        if groups and config.get("group") not in groups:
            continue

        missing = [b for b in extra_benchmarks if b not in row]
        if not missing:
            continue

        model = config["model"]
        adapter_path = None
        if config.get("training", {}).get("strategy", "none") != "none":
            p = ADAPTERS_DIR / eid
            if p.exists():
                adapter_path = str(p)
            # adapter might be deleted — still run base model eval if no adapter

        print(f"[extra_eval] {eid}: running {missing}")
        for benchmark in missing:
            try:
                problems = load_benchmark(benchmark)
                rollouts = eval_generate(
                    base_model=model,
                    lora_path=adapter_path,
                    benchmark=benchmark,
                    n=n_eval,
                    temperature=temperature,
                )
                scores = score_answers(rollouts, problems, benchmark=benchmark)
                row[benchmark] = scores
                row["timestamp"] = datetime.utcnow().isoformat()
                print(f"[extra_eval] {eid} {benchmark}: {scores}")
                updated = True
            except Exception as e:
                print(f"[extra_eval] {eid} {benchmark} FAILED: {e}")

        rows[i] = row

    if updated:
        save_results(rows)
        print("[extra_eval] results.json updated")
    else:
        print("[extra_eval] Nothing to update")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+", default=["gpqa_diamond", "livecodebench"])
    parser.add_argument("--groups", nargs="+", default=None)
    parser.add_argument("--id", default=None)
    parser.add_argument("--n-eval", type=int, default=16)
    args = parser.parse_args()

    run_extra_evals(
        extra_benchmarks=args.benchmarks,
        groups=args.groups,
        only_id=args.id,
        n_eval=args.n_eval,
    )
