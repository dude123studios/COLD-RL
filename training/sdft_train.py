"""
SDFT (Self-Distillation Fine-Tuning) training.

Trains with forward KL divergence against saved teacher logits:
  L(θ) = -Σ_t Σ_{k ∈ topK} teacher_prob[k] · log πθ(token_k | y<t, x)

The student model (with LoRA) is trained so its output distribution at each
token position matches the teacher's distribution (same model + solution context).

Dataset format (from sdft_collect.py):
  cache_dir/index.jsonl          — {id, T, answer} per problem
  cache_dir/{id}_tokens.npy      — int32 [T]  student tokens
  cache_dir/{id}_vals.npy        — float16 [T, K]  teacher logit values
  cache_dir/{id}_ids.npy         — int32 [T, K]  teacher logit token ids
"""

import gc
import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

ADAPTER_DIR = Path(__file__).parent.parent / "results" / "adapters"

SYSTEM_MSG = (
    "You are an expert math tutor. When solving problems, think out loud: "
    "explore the problem structure, consider multiple approaches, check your "
    "work as you go, and explain your reasoning at every step."
)

STUDENT_USER_SUFFIX = (
    "\n\nThink through this carefully. Explore the problem, consider different "
    "approaches, and show your full reasoning before giving the final answer."
)

LORA_TARGETS = {
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "default": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def _get_lora_targets(model) -> list[str]:
    model_type = getattr(model.config, "model_type", "default").lower()
    for key in LORA_TARGETS:
        if key in model_type:
            return LORA_TARGETS[key]
    return LORA_TARGETS["default"]


class SDFTDataset(TorchDataset):
    """
    Loads per-problem numpy arrays and tokenizes student prompts.
    Returns tensors aligned so that at each generation step the model's logits
    can be compared against the teacher's saved distribution.
    """

    def __init__(self, cache_dir: str, tokenizer, max_length: int = 8192):
        self.cache_dir = Path(cache_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length

        index_path = self.cache_dir / "index.jsonl"
        self.records = []
        with open(index_path) as f:
            for line in f:
                try:
                    self.records.append(json.loads(line))
                except Exception:
                    pass

        # Precompute student prompt prefix ids (same for all problems — just system+user)
        # We use a placeholder problem to measure prefix length; actual prefix built per item.
        print(f"[SDFT] Dataset: {len(self.records)} problems from {cache_dir}")

    def __len__(self) -> int:
        return len(self.records)

    def _student_prefix(self, problem_text: str) -> list[int]:
        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": problem_text + STUDENT_USER_SUFFIX},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer(
            text, add_special_tokens=False
        )["input_ids"]

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        pid = rec["id"]
        problem_text = rec.get("problem", "")

        student_tokens = np.load(self.cache_dir / f"{pid}_tokens.npy")  # [T]
        teacher_vals = np.load(self.cache_dir / f"{pid}_vals.npy")       # [T, K]
        teacher_ids = np.load(self.cache_dir / f"{pid}_ids.npy")         # [T, K]

        T = len(student_tokens)

        # Build real student prefix from problem text
        prefix_ids = self._student_prefix(problem_text)
        P = len(prefix_ids)

        input_ids = prefix_ids + student_tokens.tolist()
        T_use = min(T, self.max_length - P)
        if T_use <= 0:
            T_use = 1

        input_ids = input_ids[: P + T_use]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "teacher_vals": torch.tensor(
                teacher_vals[:T_use].astype(np.float32), dtype=torch.float32
            ),  # [T_use, K]
            "teacher_ids": torch.tensor(
                teacher_ids[:T_use].astype(np.int64), dtype=torch.long
            ),  # [T_use, K]
            "prompt_len": torch.tensor(P, dtype=torch.long),
        }


def _sdft_collate(batch: list[dict]) -> dict:
    """Pad variable-length sequences in a batch."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    max_T = max(b["teacher_vals"].shape[0] for b in batch)
    K = batch[0]["teacher_vals"].shape[1]

    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    teacher_vals = torch.zeros(len(batch), max_T, K, dtype=torch.float32)
    teacher_ids = torch.zeros(len(batch), max_T, K, dtype=torch.long)
    prompt_lens = torch.zeros(len(batch), dtype=torch.long)
    T_lens = torch.zeros(len(batch), dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        T = b["teacher_vals"].shape[0]
        input_ids[i, :L] = b["input_ids"]
        attention_mask[i, :L] = b["attention_mask"]
        teacher_vals[i, :T] = b["teacher_vals"]
        teacher_ids[i, :T] = b["teacher_ids"]
        prompt_lens[i] = b["prompt_len"]
        T_lens[i] = T

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "teacher_vals": teacher_vals,
        "teacher_ids": teacher_ids,
        "prompt_lens": prompt_lens,
        "T_lens": T_lens,
    }


class SDFTTrainer(Trainer):
    """Trainer that computes forward KL loss against saved teacher logits."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher_vals = inputs.pop("teacher_vals")  # [B, T, K]
        teacher_ids = inputs.pop("teacher_ids")    # [B, T, K]
        prompt_lens = inputs.pop("prompt_lens")    # [B]
        T_lens = inputs.pop("T_lens")              # [B]

        outputs = model(**inputs)
        logits = outputs.logits  # [B, seq_len, vocab]

        B = logits.shape[0]
        loss_sum = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        n_tokens = 0

        for b in range(B):
            P = prompt_lens[b].item()
            T = T_lens[b].item()
            if T == 0:
                continue

            # Student log-probs over full vocab at each generation step
            # logits[:, P-1 : P+T-1] predicts tokens at positions P : P+T
            gen_logits = logits[b, P - 1 : P + T - 1, :]  # [T, vocab]
            student_log_probs = F.log_softmax(gen_logits, dim=-1)  # [T, vocab]

            # Teacher distribution (top-K, renormalized)
            t_vals = teacher_vals[b, :T, :]  # [T, K]
            t_ids = teacher_ids[b, :T, :]    # [T, K]

            teacher_log_probs = F.log_softmax(t_vals.float(), dim=-1)  # [T, K]
            teacher_probs = teacher_log_probs.exp()                     # [T, K]

            # Gather student log-probs at teacher's top-K positions
            student_at_topk = student_log_probs.gather(1, t_ids)  # [T, K]

            # Forward KL: -Σ_k teacher_prob[k] * log student_prob[k]
            token_loss = -(teacher_probs * student_at_topk).sum(dim=-1)  # [T]
            loss_sum = loss_sum + token_loss.sum()
            n_tokens += T

        if n_tokens == 0:
            loss = loss_sum
        else:
            loss = loss_sum / n_tokens

        return (loss, outputs) if return_outputs else loss


def train_sdft(config: dict, experiment_id: str) -> str:
    """
    Train a LoRA adapter via SDFT.

    Args:
        config: experiment config dict (from YAML), must have 'sdft_cache_dir' key
        experiment_id: unique identifier, used for output directory

    Returns:
        Path to saved LoRA adapter directory
    """
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = str(ADAPTER_DIR / experiment_id)

    if (Path(output_dir) / "adapter_model.safetensors").exists():
        print(f"[sdft] Adapter already exists, skipping: {output_dir}")
        return output_dir

    training_cfg = config.get("training", {})
    model_name = config["model"]
    cache_dir = config["sdft_cache_dir"]

    lora_r = training_cfg.get("lora_r", 16)
    lora_alpha = training_cfg.get("lora_alpha", lora_r)
    lora_dropout = training_cfg.get("lora_dropout", 0.05)
    lr = training_cfg.get("lr", 2e-5)
    epochs = training_cfg.get("epochs", 1)
    batch_size = training_cfg.get("batch_size", 1)
    grad_accum = training_cfg.get("grad_accum", 16)
    max_length = training_cfg.get("max_length", 4096)
    warmup_ratio = training_cfg.get("warmup_ratio", 0.03)

    print(f"[sdft] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[sdft] Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
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

    print(f"[sdft] Loading dataset from: {cache_dir}")
    ds = SDFTDataset(cache_dir, tokenizer, max_length=max_length)
    print(f"[sdft] Dataset size: {len(ds)} problems")

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
        save_total_limit=5,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = SDFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=_sdft_collate,
    )

    print(f"[sdft] Training {experiment_id} — {total_steps} total steps")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[sdft] Adapter saved to {output_dir}")

    del trainer, ds, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[sdft] GPU freed: {free/1e9:.1f}/{total/1e9:.1f} GiB free")

    return output_dir
