"""
finetune_on_failures.py

SFT fine-tuning on tool-use failure demonstrations, with before/after eval.

Pipeline:
  1. Load base model
  2. Evaluate on eval_tool_use (baseline)
  3. Load SFT data from generate_sft_data.py output
  4. Fine-tune with SFTTrainer (LoRA, low GPU budget)
  5. Re-evaluate on eval_tool_use
  6. Report per-category delta

Usage:
    python finetune_on_failures.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --sft_data data/sft_from_failures.jsonl \
        --output_dir checkpoints/tool_use_sft \
        --output results/finetuning_results.json
"""

from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from eval_tool_use import (
    TASKS,
    eval_tool_use,
    build_react_fn,
)
from generate_sft_data import generate as generate_sft_data

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_sft_dataset(sft_data_path: str):
    """Load JSONL file produced by generate_sft_data.py as a HuggingFace Dataset."""
    from datasets import Dataset

    records: list[dict] = []
    with open(sft_data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def _formatting_func(tokenizer):
    """Return a formatting function that applies the model's chat template."""
    def fmt(example: dict) -> str:
        messages = example["messages"]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return fmt


def _run_eval(react_fn, label: str) -> dict:
    """Run eval_tool_use and return the summary dict, printing a header."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print("=" * 60)
    return eval_tool_use(react_fn=react_fn, tasks=TASKS)


def _per_category_success(summary: dict) -> dict[str, float]:
    return {
        cat: m["accuracy"]
        for cat, m in summary["per_category"].items()
    }


def _compute_delta(before: dict, after: dict) -> dict:
    """Compute per-category and overall delta between two eval summaries."""
    cats_before = _per_category_success(before)
    cats_after = _per_category_success(after)
    all_cats = sorted(set(cats_before) | set(cats_after))
    per_cat_delta = {
        cat: round(cats_after.get(cat, 0.0) - cats_before.get(cat, 0.0), 4)
        for cat in all_cats
    }
    return {
        "task_success": round(after["task_success"] - before["task_success"], 4),
        "per_category": per_cat_delta,
    }


def _print_comparison_table(before: dict, after: dict, delta: dict) -> None:
    header = f"{'Category':<22}  {'Before':>8}  {'After':>8}  {'Delta':>8}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    all_cats = sorted(
        set(_per_category_success(before)) | set(_per_category_success(after))
    )
    for cat in all_cats:
        b = _per_category_success(before).get(cat, float("nan"))
        a = _per_category_success(after).get(cat, float("nan"))
        d = delta["per_category"].get(cat, float("nan"))
        sign = "+" if d > 0 else ""
        print(f"  {cat:<20}  {b:>7.1%}  {a:>7.1%}  {sign}{d:>+.1%}")
    print(sep)
    d_overall = delta["task_success"]
    sign = "+" if d_overall > 0 else ""
    print(
        f"  {'OVERALL':<20}  {before['task_success']:>7.1%}  "
        f"{after['task_success']:>7.1%}  {sign}{d_overall:>+.1%}"
    )
    print(sep)


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def finetune(
    model_name: str,
    sft_data_path: str,
    output_dir: str,
    output_results_path: str,
    dry_run: bool = False,
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_seq_length: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    wandb_project: str = "tool-use-finetuning",
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model
    from trl import SFTTrainer, SFTConfig

    sft_path = Path(sft_data_path)
    if not sft_path.exists():
        print(f"[info] {sft_data_path} not found — generating from results/tool_use_results.json ...")
        stats = generate_sft_data(
            input_path="results/tool_use_results.json",
            output_path=sft_data_path,
            augment=1,
        )
        total_generated = sum(v for k, v in stats.items() if k != "skipped_successful")
        print(f"[info] generated {total_generated} SFT examples")

    dataset = _load_sft_dataset(sft_data_path)
    n_examples = len(dataset)
    print(f"\nLoaded {n_examples} SFT examples from {sft_data_path}")

    if dry_run:
        print("\n[dry_run] Would fine-tune with the following config:")
        print(f"  model            : {model_name}")
        print(f"  n_examples       : {n_examples}")
        print(f"  epochs           : {epochs}")
        print(f"  lr               : {lr}")
        print(f"  batch_size       : {batch_size}")
        print(f"  grad_accum_steps : {grad_accum}")
        print(f"  lora_r           : {lora_r}")
        print(f"  lora_alpha       : {lora_alpha}")
        print(f"  output_dir       : {output_dir}")
        print(f"  wandb_project    : {wandb_project}")
        steps_per_epoch = math.ceil(n_examples / (batch_size * grad_accum))
        total_steps = steps_per_epoch * epochs
        print(f"  steps_per_epoch  : {steps_per_epoch}")
        print(f"  total_steps      : {total_steps}")
        print("\n[dry_run] Skipping model load, eval, and training.")
        return {
            "model": model_name,
            "n_training_examples": n_examples,
            "dry_run": True,
            "training_config": {
                "epochs": epochs, "lr": lr, "batch_size": batch_size,
                "grad_accum": grad_accum, "lora_r": lora_r, "lora_alpha": lora_alpha,
            },
        }

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load base model (4-bit quantised for low GPU budget) ──────────────────
    print(f"Loading base model: {model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.use_cache = False

    # ── Baseline eval ─────────────────────────────────────────────────────────
    base_react_fn = build_react_fn(base_model, tokenizer)
    before_summary = _run_eval(base_react_fn, f"baseline ({model_name})")

    # ── LoRA config ───────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # ── W&B ───────────────────────────────────────────────────────────────────
    try:
        import wandb
        wandb.init(
            project=wandb_project,
            name=f"sft-tool-use-{model_name.split('/')[-1]}",
            config={
                "model": model_name,
                "n_examples": n_examples,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "before_task_success": before_summary["task_success"],
            },
        )
    except Exception as e:
        print(f"[warn] W&B init failed: {e} — continuing without logging")

    # ── SFTConfig and SFTTrainer ──────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fmt = _formatting_func(tokenizer)

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to="wandb",
        max_seq_length=max_seq_length,
        dataset_text_field=None,
    )

    trainer = SFTTrainer(
        model=base_model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=fmt,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting fine-tuning...")
    t0 = time.time()
    trainer.train()
    train_duration = round(time.time() - t0, 1)
    print(f"Training completed in {train_duration}s")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    # ── Post-training eval ────────────────────────────────────────────────────
    ft_model = trainer.model
    ft_react_fn = build_react_fn(ft_model, tokenizer)
    after_summary = _run_eval(ft_react_fn, f"fine-tuned ({output_dir})")

    # ── Compute delta and build results ───────────────────────────────────────
    delta = _compute_delta(before_summary, after_summary)

    training_config = {
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_target_modules": ["q_proj", "v_proj"],
        "bf16": True,
        "gradient_checkpointing": True,
        "max_seq_length": max_seq_length,
        "train_duration_s": train_duration,
    }

    results = {
        "model": model_name,
        "n_training_examples": n_examples,
        "before": {
            "task_success": before_summary["task_success"],
            "per_category": _per_category_success(before_summary),
        },
        "after": {
            "task_success": after_summary["task_success"],
            "per_category": _per_category_success(after_summary),
        },
        "delta": delta,
        "training_config": training_config,
    }

    Path(output_results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {output_results_path}")

    # ── Print comparison table ────────────────────────────────────────────────
    _print_comparison_table(before_summary, after_summary, delta)

    # ── Log final metrics to W&B ──────────────────────────────────────────────
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({
                "after_task_success": after_summary["task_success"],
                "delta_task_success": delta["task_success"],
            })
            wandb.finish()
    except Exception:
        pass

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a model on tool-use failure demonstrations with before/after eval."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--sft_data",
        default="data/sft_from_failures.jsonl",
        help="JSONL file from generate_sft_data.py (auto-generated if missing)",
    )
    parser.add_argument(
        "--output_dir",
        default="checkpoints/tool_use_sft",
        help="Directory to save fine-tuned model checkpoints",
    )
    parser.add_argument(
        "--output",
        default="results/finetuning_results.json",
        help="Path to write the before/after results JSON",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for training",
    )
    parser.add_argument(
        "--wandb_project",
        default="tool-use-finetuning",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would happen without loading the model or training",
    )
    args = parser.parse_args()

    results = finetune(
        model_name=args.model,
        sft_data_path=args.sft_data,
        output_dir=args.output_dir,
        output_results_path=args.output,
        dry_run=args.dry_run,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        wandb_project=args.wandb_project,
    )

    if not results.get("dry_run"):
        print(f"\nDone. Delta task_success: {results['delta']['task_success']:+.1%}")


if __name__ == "__main__":
    main()
