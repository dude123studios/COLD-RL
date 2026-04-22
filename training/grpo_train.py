"""
GRPO training using TRL's GRPOTrainer.
Can initialize from a pre-trained SFT LoRA adapter (for PAS → GRPO pipeline).
Uses vLLM sidecar for fast rollout generation.
"""

import json
import math
import os
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from generation.rollout import extract_answer
from training.sft_train import ADAPTER_DIR, _get_lora_targets

MATH500_PATH = Path(__file__).parent.parent / "data" / "math500.jsonl"


def math_reward(completions: list[str], ground_truths: list[str], **kwargs) -> list[float]:
    """Binary reward: 1.0 if extracted answer matches ground truth, else 0.0."""
    return [
        1.0 if extract_answer(c) == gt else 0.0
        for c, gt in zip(completions, ground_truths)
    ]


def _load_grpo_dataset(dataset_path: str) -> Dataset:
    """Load problems for GRPO — only needs problem + answer (no solution)."""
    rows = []
    with open(dataset_path) as f:
        for line in f:
            row = json.loads(line)
            rows.append(
                {
                    "prompt": row["input"],
                    "ground_truth": row.get("answer", ""),
                }
            )
    # Deduplicate by prompt
    seen = set()
    deduped = []
    for r in rows:
        if r["prompt"] not in seen:
            seen.add(r["prompt"])
            deduped.append(r)
    return Dataset.from_list(deduped)


def train_grpo(config: dict, experiment_id: str) -> str:
    """
    Train a GRPO LoRA adapter.

    Args:
        config: experiment config dict
        experiment_id: unique identifier

    Returns:
        Path to saved LoRA adapter directory
    """
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = str(ADAPTER_DIR / experiment_id)

    training_cfg = config.get("training", {})
    model_name = config["model"]
    dataset_path = config["dataset_path"]

    # GRPO-specific params
    lora_r = training_cfg.get("lora_r", 16)
    lora_alpha = training_cfg.get("lora_alpha", 32)
    lr = float(training_cfg.get("lr", 5e-7))
    epochs = training_cfg.get("epochs", 1)
    batch_size = training_cfg.get("batch_size", 4)
    grad_accum = training_cfg.get("grad_accum", 4)
    num_generations = training_cfg.get("num_generations", 8)
    max_prompt_length = training_cfg.get("max_prompt_length", 512)
    max_completion_length = training_cfg.get("max_completion_length", 4096)
    init_from_lora = training_cfg.get("init_from_lora")  # optional: SFT adapter path

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Optionally initialize from an SFT LoRA (PAS → GRPO pipeline)
    if init_from_lora:
        print(f"[grpo] Merging SFT LoRA from {init_from_lora}")
        model = PeftModel.from_pretrained(model, init_from_lora)
        model = model.merge_and_unload()
        print("[grpo] Merged SFT LoRA into base weights")

    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=_get_lora_targets(model),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    ds = _load_grpo_dataset(dataset_path)
    print(f"[grpo] Dataset: {len(ds)} unique problems")

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.5,
        vllm_max_model_length=max_prompt_length + max_completion_length,
        generation_kwargs={"temperature": 0.9},
    )

    def reward_fn(completions, prompts, ground_truth, **kwargs):
        return math_reward(completions, ground_truth)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=ds,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    print(f"[grpo] Training {experiment_id}")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[grpo] Adapter saved to {output_dir}")

    return output_dir
