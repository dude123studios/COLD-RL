"""
SFT training with LoRA.

Accepts a config dict and experiment_id. All hyperparameters live in YAML.
Saves only the LoRA adapter (~50-300MB), not the full model.
Intermediate checkpoints saved at 25/50/75% for R-Div-over-training plots.
"""

import gc
import json
import math
import os
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)

ADAPTER_DIR = Path(__file__).parent.parent / "results" / "adapters"

# LoRA target modules by model family
LORA_TARGETS = {
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "default": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def _chat_end_token(tokenizer) -> str:
    """Return the correct chat-turn end token for the loaded tokenizer."""
    vocab = tokenizer.get_vocab()
    if "<|im_end|>" in vocab:
        return "<|im_end|>"        # Qwen2/2.5
    if "<|eot_id|>" in vocab:
        return "<|eot_id|>"        # Llama-3
    if "<end_of_turn>" in vocab:
        return "<end_of_turn>"     # Gemma
    return tokenizer.eos_token     # Mistral and fallback


def _get_lora_targets(model) -> list[str]:
    model_type = getattr(model.config, "model_type", "default").lower()
    for key in LORA_TARGETS:
        if key in model_type:
            return LORA_TARGETS[key]
    return LORA_TARGETS["default"]


def _load_dataset(dataset_path: str, tokenizer, max_length: int = 4096) -> Dataset:
    rows = []
    with open(dataset_path) as f:
        for line in f:
            rows.append(json.loads(line))

    def tokenize(row):
        prompt = row["input"]
        completion = row["output"]

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        chat_end = _chat_end_token(tokenizer)
        completion_ids = tokenizer(
            completion + chat_end, add_special_tokens=False
        )["input_ids"]

        input_ids = prompt_ids + completion_ids
        # Mask prompt tokens from loss
        labels = [-100] * len(prompt_ids) + completion_ids

        # Truncate
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    ds = Dataset.from_list(rows)
    ds = ds.map(tokenize, remove_columns=ds.column_names)
    return ds


def train_sft(config: dict, experiment_id: str) -> str:
    """
    Train a LoRA adapter via SFT.
    Skips training if a completed adapter already exists on disk.

    Args:
        config: experiment config dict (from YAML)
        experiment_id: unique identifier, used for output directory

    Returns:
        Path to saved LoRA adapter directory
    """
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = str(ADAPTER_DIR / experiment_id)

    # Skip training if a completed adapter already exists
    if (Path(output_dir) / "adapter_model.safetensors").exists():
        print(f"[sft] Adapter already exists, skipping training: {output_dir}", flush=True)
        return output_dir

    training_cfg = config.get("training", {})
    model_name = config["model"]
    dataset_path = config["dataset_path"]

    # Hyperparameters with defaults
    lora_r = training_cfg.get("lora_r", 16)
    lora_alpha = training_cfg.get("lora_alpha", lora_r)  # default scaling=1.0, not 2.0
    lora_dropout = training_cfg.get("lora_dropout", 0.05)
    lr = training_cfg.get("lr", 2e-5)
    epochs = training_cfg.get("epochs", 1)
    batch_size = training_cfg.get("batch_size", 1)
    grad_accum = training_cfg.get("grad_accum", 16)
    max_length = training_cfg.get("max_length", 4096)
    warmup_ratio = training_cfg.get("warmup_ratio", 0.03)

    print(f"[sft] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[sft] Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=_get_lora_targets(model),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()

    print(f"[sft] Loading dataset: {dataset_path}")
    ds = _load_dataset(dataset_path, tokenizer, max_length)
    print(f"[sft] Dataset size: {len(ds)} rows")

    # Compute checkpoint steps for 25/50/75/100% saves
    steps_per_epoch = math.ceil(len(ds) / (batch_size * grad_accum))
    total_steps = steps_per_epoch * epochs
    save_steps = max(1, total_steps // 4)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=5,  # keep all 4 checkpoints + final
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
    )

    print(f"[sft] Training {experiment_id} — {total_steps} total steps")
    trainer.train()

    # Save only LoRA adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[sft] Adapter saved to {output_dir}")

    # Explicitly free GPU memory before returning — the next experiment's
    # model load (or vLLM) will OOM if we leave these alive.
    del trainer, collator, ds, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[sft] GPU freed: {free/1e9:.1f}/{total/1e9:.1f} GiB free", flush=True)

    return output_dir
