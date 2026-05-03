"""
Standalone vLLM generation worker.
Called as a subprocess by eval_generate() so that vLLM's atexit/os._exit()
only kills this worker process, not the main orchestrator.

Usage (internal — called by vllm_generate.py):
  python -m evaluation.vllm_worker \
      --model MODEL \
      --prompts-file PROMPTS_JSONL \
      --out-file OUT_JSONL \
      [--lora-path PATH] \
      [--n 16] [--temperature 0.6] [--max-tokens 4096]
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    # Load prompts
    with open(args.prompts_file) as f:
        items = [json.loads(l) for l in f]

    prompts = [it["prompt"] for it in items]
    ids = [it["id"] for it in items]

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    import torch
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    gpu_util = 0.70 if total_gb >= 70 else 0.40

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_util,
        max_model_len=8192,
        enforce_eager=True,
        disable_log_stats=True,
        dtype="bfloat16",
    )
    if args.lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    lora_req = LoRARequest("adapter", 1, args.lora_path) if args.lora_path else None
    outputs = llm.generate(prompts, sampling, lora_request=lora_req)

    with open(args.out_file, "w") as f:
        for pid, output in zip(ids, outputs):
            rollouts = [o.text for o in output.outputs]
            f.write(json.dumps({"id": pid, "rollouts": rollouts}) + "\n")

    print(f"[worker] Written {len(ids)} results to {args.out_file}", flush=True)


if __name__ == "__main__":
    main()
