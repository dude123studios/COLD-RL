"""
Download and prepare benchmark data for GPQA Diamond and LiveCodeBench.

Usage:
    python -m scripts.download_benchmarks

GPQA Diamond is a gated dataset. If access is denied, visit:
    https://huggingface.co/datasets/Idavidrein/gpqa
and click "Request access". Then re-run this script.
"""

import hashlib
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "benchmarks"


def download_gpqa_diamond():
    out_path = DATA_DIR / "gpqa_diamond.jsonl"
    if out_path.exists():
        print(f"[gpqa] Already downloaded: {out_path}")
        return

    print("[gpqa] Downloading GPQA Diamond from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    except Exception as e:
        if "gated" in str(e).lower() or "403" in str(e):
            print(
                "[gpqa] ACCESS DENIED — GPQA Diamond is a gated dataset.\n"
                "  1. Visit https://huggingface.co/datasets/Idavidrein/gpqa\n"
                "  2. Click 'Request access' and agree to the terms\n"
                "  3. Run this script again once approved\n"
            )
        else:
            print(f"[gpqa] Download failed: {e}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, item in enumerate(ds):
        # Deterministically shuffle the 4 answer choices using the problem hash as seed
        qid = f"gpqa_{i:04d}"
        choices_text = [
            item["Correct Answer"],
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]
        rng = random.Random(hashlib.md5(item["Question"].encode()).hexdigest())
        indices = list(range(4))
        rng.shuffle(indices)
        shuffled = [choices_text[j] for j in indices]
        correct_idx = indices.index(0)  # where did the correct answer end up?
        correct_letter = "ABCD"[correct_idx]

        letters = "ABCD"
        choices_formatted = "\n".join(f"{letters[j]}) {shuffled[j]}" for j in range(4))

        problem = (
            f"{item['Question']}\n\n"
            f"{choices_formatted}\n\n"
            f"Think step by step, then write your final answer as a single letter "
            f"(A, B, C, or D) on the last line."
        )

        rows.append({
            "id": qid,
            "problem": problem,
            "answer": correct_letter,
            "question_raw": item["Question"],
            "choices": shuffled,
            "correct_choice_text": choices_text[0],
        })

    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[gpqa] Saved {len(rows)} problems to {out_path}")


def download_livecodebench():
    out_path = DATA_DIR / "livecodebench.jsonl"
    if out_path.exists():
        print(f"[lcb] Already downloaded: {out_path}")
        return

    print("[lcb] Downloading LiveCodeBench v6 from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download
        raw_path = hf_hub_download(
            "livecodebench/code_generation_lite", "test6.jsonl", repo_type="dataset"
        )
    except Exception as e:
        print(f"[lcb] Download failed: {e}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(raw_path) as f:
        for line in f:
            item = json.loads(line)
            public_tests = json.loads(item["public_test_cases"])

            problem = (
                f"{item['question_content']}\n\n"
                f"Write a complete Python solution. "
                f"Enclose your code in a ```python ... ``` block."
            )
            if item.get("starter_code"):
                problem += f"\n\nStarter code:\n```python\n{item['starter_code']}\n```"

            rows.append({
                "id": item["question_id"],
                "problem": problem,
                "answer": "__code__",  # sentinel: scored by execution, not string match
                "title": item["question_title"],
                "difficulty": item["difficulty"],
                "platform": item["platform"],
                "public_test_cases": public_tests,
                "starter_code": item.get("starter_code", ""),
            })

    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[lcb] Saved {len(rows)} problems to {out_path} "
          f"(easy={sum(1 for r in rows if r['difficulty']=='easy')}, "
          f"medium={sum(1 for r in rows if r['difficulty']=='medium')}, "
          f"hard={sum(1 for r in rows if r['difficulty']=='hard')})")


def download_aime24():
    out_path = DATA_DIR / "aime24.jsonl"
    if out_path.exists():
        print(f"[aime24] Already downloaded: {out_path}")
        return

    print("[aime24] Downloading AIME 2024 from HuggingFace...")
    rows = []
    for hf_name in ["qq8933/AIME_1983_2024", "Maxwell-Jia/AIME_2024", "AI-MO/aimo-validation-aime"]:
        try:
            from datasets import load_dataset
            ds = load_dataset(hf_name, split="train", trust_remote_code=True)
            for i, item in enumerate(ds):
                year = str(item.get("year") or item.get("Year") or "")
                if year and year != "2024":
                    continue
                problem = item.get("problem") or item.get("Problem") or item.get("question") or ""
                answer = item.get("answer") or item.get("Answer") or ""
                pid = item.get("id") or f"aime24_{len(rows)}"
                if not problem:
                    continue
                try:
                    answer = str(int(str(answer).strip()))
                except Exception:
                    pass
                rows.append({"id": str(pid), "problem": str(problem), "answer": str(answer)})
            if rows:
                print(f"[aime24] Got {len(rows)} problems from {hf_name}")
                break
        except Exception as e:
            print(f"[aime24] {hf_name} failed: {e}")

    if not rows:
        print("[aime24] WARNING: Could not download. Place problems manually at data/benchmarks/aime24.jsonl")
        print("[aime24] Format: {\"id\": \"aime24_0\", \"problem\": \"...\", \"answer\": \"7\"}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[aime24] Saved {len(rows)} problems to {out_path}")


def download_mmlu(n_per_subject: int = 50):
    out_path = DATA_DIR / "mmlu.jsonl"
    if out_path.exists():
        print(f"[mmlu] Already downloaded: {out_path}")
        return

    STEM_SUBJECTS = [
        "abstract_algebra", "college_chemistry", "college_mathematics",
        "college_physics", "high_school_chemistry", "high_school_mathematics",
        "high_school_physics", "high_school_biology", "college_biology",
    ]
    print("[mmlu] Downloading MMLU STEM subset from HuggingFace...")
    rows = []
    try:
        from datasets import load_dataset
        for subj in STEM_SUBJECTS:
            try:
                ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=True)
                for i, item in enumerate(ds):
                    if i >= n_per_subject:
                        break
                    choices = item.get("choices", [])
                    choice_str = "\n".join(f"({chr(65+j)}) {c}" for j, c in enumerate(choices))
                    problem = item["question"] + "\n\n" + choice_str + "\n\nWrite your final answer as a single letter (A, B, C, or D) on the last line."
                    answer = chr(65 + int(item["answer"]))
                    rows.append({"id": f"mmlu_{subj}_{i}", "problem": problem, "answer": answer, "subject": subj})
            except Exception as e:
                print(f"[mmlu]   Skipping {subj}: {e}")
    except Exception as e:
        print(f"[mmlu] Download failed: {e}")

    if not rows:
        print("[mmlu] WARNING: Could not download MMLU.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[mmlu] Saved {len(rows)} problems to {out_path}")


def download_arc():
    out_path = DATA_DIR / "arc.jsonl"
    if out_path.exists():
        print(f"[arc] Already downloaded: {out_path}")
        return

    print("[arc] Downloading ARC-Challenge from HuggingFace...")
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", trust_remote_code=True)
        for i, item in enumerate(ds):
            if i >= 300:
                break
            labels = item["choices"]["label"]
            texts = item["choices"]["text"]
            choice_str = "\n".join(f"({l}) {t}" for l, t in zip(labels, texts))
            problem = item["question"] + "\n\n" + choice_str + "\n\nWrite your final answer as a single letter on the last line."
            rows.append({"id": f"arc_{i}", "problem": problem, "answer": item["answerKey"]})
    except Exception as e:
        print(f"[arc] Download failed: {e}")

    if not rows:
        print("[arc] WARNING: Could not download ARC.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[arc] Saved {len(rows)} problems to {out_path}")


def download_gsm8k():
    out_path = DATA_DIR / "gsm8k.jsonl"
    if out_path.exists():
        print(f"[gsm8k] Already downloaded: {out_path}")
        return

    print("[gsm8k] Downloading GSM8K from HuggingFace...")
    rows = []
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        for i, item in enumerate(ds):
            # Extract just the number from "#### 42" style answers
            raw_answer = item["answer"]
            answer_line = [l for l in raw_answer.split("\n") if l.startswith("####")]
            if answer_line:
                answer = answer_line[-1].replace("####", "").strip().replace(",", "")
            else:
                answer = raw_answer.strip()
            problem = item["question"] + "\n\nSolve step by step, then put your final answer within \\boxed{}."
            rows.append({"id": f"gsm8k_{i}", "problem": problem, "answer": answer})
    except Exception as e:
        print(f"[gsm8k] Download failed: {e}")

    if not rows:
        print("[gsm8k] WARNING: Could not download GSM8K.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[gsm8k] Saved {len(rows)} problems to {out_path}")


def download_amc23():
    out_path = DATA_DIR / "amc23.jsonl"
    if out_path.exists():
        print(f"[amc23] Already downloaded: {out_path}")
        return

    print("[amc23] Downloading AMC 2023 from HuggingFace...")
    rows = []
    for hf_name in ["math-ai/AMC2023", "Rujun11/AMC-2023"]:
        try:
            from datasets import load_dataset
            ds = load_dataset(hf_name, split="train", trust_remote_code=True)
            for i, item in enumerate(ds):
                problem = item.get("problem") or item.get("question") or ""
                answer = str(item.get("answer") or item.get("solution") or "").strip()
                if not problem:
                    continue
                pid = item.get("id") or f"amc23_{len(rows)}"
                rows.append({"id": str(pid), "problem": str(problem), "answer": answer})
            if rows:
                print(f"[amc23] Got {len(rows)} problems from {hf_name}")
                break
        except Exception as e:
            print(f"[amc23] {hf_name} failed: {e}")

    if not rows:
        print("[amc23] WARNING: Could not download. Place manually at data/benchmarks/amc23.jsonl")
        print("[amc23] Format: {\"id\": \"amc23_0\", \"problem\": \"...\", \"answer\": \"7\"}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[amc23] Saved {len(rows)} problems to {out_path}")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_gpqa_diamond()
    download_livecodebench()
    download_aime24()
    download_mmlu()
    download_arc()
    download_gsm8k()
    download_amc23()
