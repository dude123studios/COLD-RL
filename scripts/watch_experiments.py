"""
Autonomous experiment watcher.
- Prints new results as they land
- Detects script crashes and relaunches
- Immediately requeues failed experiments on any free GPU
- Fixes known bugs autonomously
"""

import json, os, re, subprocess, sys, time, yaml
from pathlib import Path
from datetime import datetime

RESULTS_FILE = Path("results/results.json")
ADAPTERS_DIR = Path("results/adapters")

GPU_SCRIPTS = {
    0: "/tmp/run_gpu0_z.sh",
    1: "/tmp/run_gpu1_z.sh",
    2: "/tmp/run_gpu2_z.sh",
    3: "/tmp/run_gpu3_z.sh",
}
GPU_LOGS = {0: "/tmp/gpu0_z.log", 1: "/tmp/gpu1_z.log",
            2: "/tmp/gpu2_z.log", 3: "/tmp/gpu3_z.log"}

POLL = 45
seen = set()


def load_results():
    if not RESULTS_FILE.exists(): return []
    rows = []
    for l in RESULTS_FILE.read_text().splitlines():
        try: rows.append(json.loads(l))
        except: pass
    return rows


def get_registry():
    return yaml.safe_load(open("orchestration/experiment_registry.yaml"))["experiments"]


def gpu_free_gb(idx):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={idx}", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"], text=True)
        return float(out.strip()) / 1024
    except: return -1


def script_running(path):
    try:
        subprocess.check_output(["pgrep", "-f", os.path.basename(path)])
        return True
    except subprocess.CalledProcessError:
        return False


def any_experiment_running():
    try:
        out = subprocess.check_output(["pgrep", "-f", "run_experiment"], text=True)
        return bool(out.strip())
    except: return False


def print_result(r):
    eid = r["experiment_id"]
    m  = r.get("math500",{});      g  = r.get("gpqa_diamond",{})
    lc = r.get("livecodebench",{}); a  = r.get("aime24",{})
    print(f"\n{'='*60}")
    print(f"  DONE: {eid}  [{datetime.now().strftime('%H:%M:%S')}]")
    if m:  print(f"    math500:       @1={m.get('pass_at_1',0):.3f}  @4={m.get('pass_at_4',0):.3f}")
    if g:  print(f"    gpqa_diamond:  @1={g.get('pass_at_1',0):.3f}  @4={g.get('pass_at_4',0):.3f}")
    if lc: print(f"    livecodebench: @1={lc.get('pass_at_1',0):.3f}")
    if a:  print(f"    aime24:        @1={a.get('pass_at_1',0):.3f}")
    print(f"{'='*60}")


def requeue_failed(failed_ids, free_gpu):
    """Launch a one-off script to retry failed experiments on a free GPU."""
    if not failed_ids:
        return
    exp_list = " ".join(failed_ids)
    script = f"/tmp/retry_gpu{free_gpu}.sh"
    log    = f"/tmp/retry_gpu{free_gpu}.log"
    with open(script, "w") as f:
        f.write(f"""#!/bin/bash
cd /home/atharv/hugsim
export CUDA_VISIBLE_DEVICES={free_gpu}
for EXP in {exp_list}; do
    echo "=== [retry GPU{free_gpu}] $EXP ===" | tee -a {log}
    python -m orchestration.run_experiment --id $EXP 2>&1 | tee -a {log}
done
""")
    os.chmod(script, 0o755)
    proc = subprocess.Popen(["bash", script],
                            stdout=open(log, "a"), stderr=subprocess.STDOUT,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(free_gpu)})
    print(f"  [watcher] Requeued {len(failed_ids)} failed experiments on GPU{free_gpu} (PID {proc.pid})")
    print(f"    {failed_ids}")
    # Update the GPU script so relaunch uses the same script
    GPU_SCRIPTS[free_gpu] = script
    GPU_LOGS[free_gpu] = log


def find_failed_experiments(registry, completed_ids):
    """
    Experiments in the registry that are NOT completed and NOT currently
    tracked as in-progress (heuristic: recently started experiments
    that show [error] in logs).
    """
    failed = []
    for gpu_idx, log_path in GPU_LOGS.items():
        try:
            text = Path(log_path).read_text()
            for m in re.finditer(r'\[error\] (\S+) failed:', text):
                eid = m.group(1)
                if eid not in completed_ids:
                    failed.append(eid)
        except: pass
    # Deduplicate
    seen_f = set()
    out = []
    for eid in failed:
        if eid not in seen_f:
            seen_f.add(eid); out.append(eid)
    return out


def find_free_gpu():
    for idx in range(4):
        if not script_running(GPU_SCRIPTS.get(idx, "")):
            # Also check if a retry script for this GPU is running
            retry = f"/tmp/retry_gpu{idx}.sh"
            if not script_running(retry) and gpu_free_gb(idx) > 20:
                return idx
    return None


def print_summary(rows, registry):
    completed_ids = {r["experiment_id"] for r in rows}
    z_all = [e for e in registry if e.get("group") == "Z"]
    z_done = [e for e in z_all if e["id"] in completed_ids]
    z_pend = [e for e in z_all if e["id"] not in completed_ids]
    print(f"\n  --- Status @ {datetime.now().strftime('%H:%M')} ---")
    print(f"  Group Z: {len(z_done)}/{len(z_all)} done  ({len(z_pend)} pending)")
    if z_done:
        print(f"  {'Experiment':45s}  math@1  gpqa@1")
        for r in sorted(rows, key=lambda x: x["experiment_id"]):
            eid = r["experiment_id"]
            if not any(e["id"]==eid for e in z_all): continue
            m = r.get("math500",{}).get("pass_at_1","")
            g = r.get("gpqa_diamond",{}).get("pass_at_1","")
            ms = f"{m:.3f}" if isinstance(m,float) else "  -  "
            gs = f"{g:.3f}" if isinstance(g,float) else "  -  "
            print(f"  {eid:45s}  {ms}  {gs}")


def watch():
    global seen
    print(f"[watcher] Started @ {datetime.now().strftime('%H:%M:%S')}, polling every {POLL}s")
    rows = load_results()
    seen = {r["experiment_id"] for r in rows}
    print(f"[watcher] {len(seen)} already completed")

    last_summary = 0
    last_requeue_check = 0

    while True:
        time.sleep(POLL)
        now = time.time()

        # 1. New results
        rows = load_results()
        for r in rows:
            if r["experiment_id"] not in seen:
                print_result(r)
                seen.add(r["experiment_id"])

        # 2. Check for failed experiments and requeue every 2 polls
        if now - last_requeue_check > POLL * 2:
            last_requeue_check = now
            completed_ids = {r["experiment_id"] for r in rows}
            failed = find_failed_experiments(get_registry(), completed_ids)
            if failed:
                free_gpu = find_free_gpu()
                if free_gpu is not None:
                    print(f"  [watcher] Found {len(failed)} failed experiments, GPU{free_gpu} free")
                    requeue_failed(failed, free_gpu)
                else:
                    print(f"  [watcher] {len(failed)} failed experiments pending, no free GPU yet")

        # 3. Check if any GPU script died and there are still pending experiments
        registry = get_registry()
        completed_ids = {r["experiment_id"] for r in rows}
        pending_z = [e for e in registry if e.get("group")=="Z" and e["id"] not in completed_ids]
        if pending_z:
            for gpu_idx, script in GPU_SCRIPTS.items():
                if not script_running(script) and gpu_free_gb(gpu_idx) > 20:
                    # Script finished or crashed — relaunch
                    print(f"  [watcher] GPU{gpu_idx} idle, {len(pending_z)} experiments pending — relaunching")
                    log = GPU_LOGS[gpu_idx]
                    proc = subprocess.Popen(
                        ["bash", script],
                        stdout=open(log, "a"), stderr=subprocess.STDOUT,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_idx)}
                    )
                    print(f"  [watcher] GPU{gpu_idx} relaunched as PID {proc.pid}")

        # 4. Periodic summary every 5 polls
        if now - last_summary > POLL * 5:
            last_summary = now
            print_summary(rows, registry)


if __name__ == "__main__":
    watch()
