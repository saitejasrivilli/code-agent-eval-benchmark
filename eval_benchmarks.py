#!/usr/bin/env python3
"""
Multi-benchmark evaluation harness for fine-tuned LLMs.

Covers three evaluation axes:
  1. Math reasoning  — GSM8K (verifiable exact-match accuracy)
  2. Code generation — HumanEval-style (execute and check test cases)
  3. LLM-as-judge   — open-ended quality scoring (0–10, no ground truth required)

Outputs a structured JSON report + a markdown summary table.

Usage:
  python 6_evaluation/eval_benchmarks.py \
      --base_model Qwen/Qwen2.5-7B-Instruct \
      --adapter_path models/grpo_policy/best \
      --benchmarks gsm8k humaneval llm_judge \
      --gsm8k_n 100 --humaneval_n 40 --judge_n 20 \
      --output_dir results/eval

Requires:
  pip install datasets transformers peft torch
  # For HumanEval: pip install human-eval  (optional — fallback available)
"""
import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(base_model: str, adapter_path: Optional[str] = None):
    logger.info("Loading base model: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        cache_dir="/storage/gxg8313/saiteja/hf_cache",
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/storage/gxg8313/saiteja/hf_cache",
    )
    if adapter_path and Path(adapter_path).exists():
        logger.info("Loading adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new: int = 512, temp: float = 0.0) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=(temp > 0),
            temperature=temp or 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


# ── GSM8K Evaluation ──────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _extract_number(text: str) -> Optional[float]:
    text = text.lower().replace(",", "")
    # "the answer is X" / "= X"
    for pat in [r"answer[^=\n]*?=\s*(-?\d+\.?\d*)",
                r"####\s*(-?\d+\.?\d*)",
                r"=\s*(-?\d+\.?\d*)\s*$"]:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    nums = _NUM_RE.findall(text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def eval_gsm8k(model, tokenizer, n: int = 100) -> Dict:
    """Evaluate on GSM8K test split (exact-match numeric accuracy)."""
    from datasets import load_dataset
    logger.info("Evaluating GSM8K (n=%d)...", n)
    ds = load_dataset("gsm8k", "main", split="test",
                      cache_dir="/storage/gxg8313/saiteja/hf_cache")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))

    correct = 0
    results = []

    for row in ds:
        prompt = (
            "<|im_start|>system\nSolve the math problem step by step. "
            "End your answer with '#### NUMBER'.<|im_end|>\n"
            f"<|im_start|>user\n{row['question']}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        pred_text = generate(model, tokenizer, prompt, max_new=256)
        gt_match  = re.search(r"####\s*([\d,.-]+)", row["answer"])
        gt_val    = float(gt_match.group(1).replace(",","")) if gt_match else None
        pred_val  = _extract_number(pred_text)

        hit = (gt_val is not None and pred_val is not None and
               abs(pred_val - gt_val) < max(1e-4, abs(gt_val) * 1e-4))
        correct += int(hit)
        results.append({"question": row["question"][:80], "gt": gt_val,
                         "pred": pred_val, "correct": hit})

    accuracy = correct / len(ds)
    logger.info("GSM8K accuracy: %.3f (%d/%d)", accuracy, correct, len(ds))
    return {"benchmark": "gsm8k", "n": len(ds), "accuracy": round(accuracy, 4),
            "correct": correct, "samples": results[:5]}


# ── HumanEval-style Code Evaluation ──────────────────────────────────────────

_CODE_PROBLEMS = [
    {
        "id": "he_1", "language": "python",
        "prompt": "def has_close_elements(numbers: list, threshold: float) -> bool:\n    \"\"\"Check if any two numbers are closer than threshold.\"\"\"\n",
        "test": "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\nassert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False",
        "entry_point": "has_close_elements",
    },
    {
        "id": "he_2", "language": "python",
        "prompt": "def separate_paren_groups(paren_string: str) -> list:\n    \"\"\"Separate top-level parenthesized groups.\"\"\"\n",
        "test": "assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']",
        "entry_point": "separate_paren_groups",
    },
    {
        "id": "he_3", "language": "python",
        "prompt": "def truncate_number(number: float) -> float:\n    \"\"\"Return decimal part of number.\"\"\"\n",
        "test": "assert truncate_number(3.5) == 0.5\nassert truncate_number(1.33) == 0.33000000000000007",
        "entry_point": "truncate_number",
    },
    {
        "id": "he_4", "language": "python",
        "prompt": "def below_zero(operations: list) -> bool:\n    \"\"\"Check if balance ever goes below zero.\"\"\"\n",
        "test": "assert below_zero([1, 2, 3]) == False\nassert below_zero([1, 2, -4, 5]) == True",
        "entry_point": "below_zero",
    },
    {
        "id": "he_5", "language": "python",
        "prompt": "def mean_absolute_deviation(numbers: list) -> float:\n    \"\"\"MAD from mean.\"\"\"\n",
        "test": "assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6",
        "entry_point": "mean_absolute_deviation",
    },
    {
        "id": "he_6", "language": "python",
        "prompt": "def intersperse(numbers: list, delimeter: int) -> list:\n    \"\"\"Intersperse delimeter between numbers.\"\"\"\n",
        "test": "assert intersperse([], 4) == []\nassert intersperse([1, 2, 3], 4) == [1, 4, 2, 4, 3]",
        "entry_point": "intersperse",
    },
    {
        "id": "he_7", "language": "python",
        "prompt": "def parse_nested_parens(paren_string: str) -> list:\n    \"\"\"Return max nesting depth for each space-separated group.\"\"\"\n",
        "test": "assert parse_nested_parens('(()()) ((())) () ((())()())') == [2, 3, 1, 3]",
        "entry_point": "parse_nested_parens",
    },
    {
        "id": "he_8", "language": "python",
        "prompt": "def filter_by_substring(strings: list, substring: str) -> list:\n    \"\"\"Filter strings containing substring.\"\"\"\n",
        "test": "assert filter_by_substring([], 'a') == []\nassert filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a') == ['abc', 'bacd', 'array']",
        "entry_point": "filter_by_substring",
    },
    {
        "id": "he_9", "language": "python",
        "prompt": "def sum_product(numbers: list) -> tuple:\n    \"\"\"Return (sum, product) of numbers.\"\"\"\n",
        "test": "assert sum_product([]) == (0, 1)\nassert sum_product([1, 2, 3, 4]) == (10, 24)",
        "entry_point": "sum_product",
    },
    {
        "id": "he_10", "language": "python",
        "prompt": "def rolling_max(numbers: list) -> list:\n    \"\"\"Running max of input list.\"\"\"\n",
        "test": "assert rolling_max([1, 2, 3, 2, 3, 4, 2]) == [1, 2, 3, 3, 3, 4, 4]",
        "entry_point": "rolling_max",
    },
]


def _exec_code(code: str, timeout: int = 5) -> Tuple[bool, str]:
    """Execute code in subprocess, return (passed, stderr)."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0, result.stderr[:200]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def _extract_code(text: str, prompt: str) -> str:
    """Extract Python function body from generated text."""
    # Try ```python ... ```
    m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try def ... block
    m = re.search(r"(def \w+\(.*?(?=\ndef |\Z))", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def eval_humaneval(model, tokenizer, n: int = 40) -> Dict:
    """Evaluate on HumanEval-style code problems (functional correctness)."""
    problems = _CODE_PROBLEMS[:min(n, len(_CODE_PROBLEMS))]
    logger.info("Evaluating HumanEval-style (%d problems)...", len(problems))

    passed = 0
    results = []

    for prob in problems:
        prompt_text = (
            "<|im_start|>system\nComplete the Python function. "
            "Only output the function body, no explanation.<|im_end|>\n"
            f"<|im_start|>user\n```python\n{prob['prompt']}```\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        completion = generate(model, tokenizer, prompt_text, max_new=256)
        code_body  = _extract_code(completion, prob["prompt"])
        full_code  = prob["prompt"] + "\n" + code_body + "\n\n" + prob["test"]
        ok, err    = _exec_code(full_code)
        passed    += int(ok)
        results.append({"id": prob["id"], "passed": ok, "error": err if not ok else ""})

    pass_at_1 = passed / max(len(problems), 1)
    logger.info("HumanEval pass@1: %.3f (%d/%d)", pass_at_1, passed, len(problems))
    return {"benchmark": "humaneval", "n": len(problems), "pass_at_1": round(pass_at_1, 4),
            "passed": passed, "samples": results}


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────

_JUDGE_QUESTIONS = [
    {"category": "math_reasoning",
     "prompt": "What is 15% of 840? Show your work."},
    {"category": "math_reasoning",
     "prompt": "A train travels at 60 mph for 2.5 hours then 80 mph for 1.5 hours. What is the average speed?"},
    {"category": "code",
     "prompt": "Write a Python function that takes a list of integers and returns the second-largest element."},
    {"category": "code",
     "prompt": "Write a Python function to check if a string is a valid palindrome, ignoring spaces and case."},
    {"category": "reasoning",
     "prompt": "If all programmers drink coffee, and Bob drinks coffee, is Bob necessarily a programmer? Explain."},
    {"category": "reasoning",
     "prompt": "A farmer has 17 sheep. All but 9 die. How many sheep are left?"},
    {"category": "instruction_following",
     "prompt": "List 5 sorting algorithms in order of worst-case time complexity from best to worst."},
    {"category": "instruction_following",
     "prompt": "Explain the difference between process and thread in exactly 3 sentences."},
    {"category": "knowledge",
     "prompt": "Explain what gradient descent is and why we need learning rate scheduling."},
    {"category": "knowledge",
     "prompt": "What is the difference between precision and recall in classification?"},
]

_JUDGE_PROMPT = """Evaluate this AI response on a scale of 1-10.

Question: {question}

Response: {response}

Score based on:
- Correctness (is the answer right?)
- Completeness (does it fully address the question?)
- Clarity (is the explanation clear and well-structured?)

Reply with ONLY: Score: X/10
Then one sentence explaining the main strength or weakness."""


def eval_llm_judge(model, tokenizer, n: int = 20) -> Dict:
    """Self-judge evaluation: model scores its own responses."""
    questions = _JUDGE_QUESTIONS[:min(n, len(_JUDGE_QUESTIONS))]
    logger.info("Running LLM-as-judge (%d questions)...", len(questions))
    scores = []
    results = []

    for q in questions:
        # Generate response
        resp_prompt = (
            "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
            f"<|im_start|>user\n{q['prompt']}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        response = generate(model, tokenizer, resp_prompt, max_new=300)

        # Judge the response
        judge_prompt = (
            "<|im_start|>system\nYou are an expert evaluator.<|im_end|>\n"
            "<|im_start|>user\n"
            + _JUDGE_PROMPT.format(question=q["prompt"], response=response[:600])
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        judgment = generate(model, tokenizer, judge_prompt, max_new=80)

        # Extract score
        m = re.search(r"score:\s*(\d+(?:\.\d+)?)\s*/\s*10", judgment, re.IGNORECASE)
        score = float(m.group(1)) / 10.0 if m else 0.5
        scores.append(score)
        results.append({
            "category": q["category"],
            "question": q["prompt"][:60],
            "score_10": round(score * 10, 1),
            "judgment": judgment[:150],
        })

    avg_score = sum(scores) / max(len(scores), 1)
    by_cat = {}
    for r in results:
        cat = r["category"]
        by_cat.setdefault(cat, []).append(r["score_10"])
    cat_avgs = {k: round(sum(v) / len(v), 2) for k, v in by_cat.items()}

    logger.info("LLM-judge avg score: %.2f/10 | by category: %s",
                avg_score * 10, cat_avgs)
    return {"benchmark": "llm_judge", "n": len(questions),
            "avg_score_10": round(avg_score * 10, 2),
            "by_category": cat_avgs, "samples": results}


# ── Report ────────────────────────────────────────────────────────────────────

def generate_report(results: Dict, out_dir: Path):
    """Write JSON + markdown summary."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Markdown table
    lines = ["# Evaluation Report\n",
             f"Model: `{results['model']}`  \n",
             f"Adapter: `{results.get('adapter', 'none')}`  \n",
             f"Date: {results['timestamp']}\n\n",
             "## Summary\n",
             "| Benchmark | Metric | Score |",
             "|-----------|--------|-------|",
             ]

    for bname, bres in results["benchmarks"].items():
        if bname == "gsm8k":
            lines.append(f"| GSM8K (n={bres['n']}) | Accuracy | {bres['accuracy']:.3f} |")
        elif bname == "humaneval":
            lines.append(f"| HumanEval (n={bres['n']}) | pass@1 | {bres['pass_at_1']:.3f} |")
        elif bname == "llm_judge":
            lines.append(f"| LLM-as-Judge (n={bres['n']}) | Avg score | {bres['avg_score_10']:.1f}/10 |")
            for cat, avg in bres.get("by_category", {}).items():
                lines.append(f"| &nbsp;&nbsp;↳ {cat} | score | {avg:.1f}/10 |")

    md = "\n".join(lines)
    with open(out_dir / "eval_report.md", "w") as f:
        f.write(md)
    print(md)
    return md


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",   default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--benchmarks",   nargs="+",
                   choices=["gsm8k", "humaneval", "llm_judge"],
                   default=["gsm8k", "humaneval", "llm_judge"])
    p.add_argument("--gsm8k_n",     type=int, default=100)
    p.add_argument("--humaneval_n", type=int, default=10)
    p.add_argument("--judge_n",     type=int, default=10)
    p.add_argument("--output_dir",  default="results/eval")
    args = p.parse_args()

    model, tokenizer = load_model(args.base_model, args.adapter_path)

    bench_results = {}
    if "gsm8k" in args.benchmarks:
        bench_results["gsm8k"] = eval_gsm8k(model, tokenizer, args.gsm8k_n)
    if "humaneval" in args.benchmarks:
        bench_results["humaneval"] = eval_humaneval(model, tokenizer, args.humaneval_n)
    if "llm_judge" in args.benchmarks:
        bench_results["llm_judge"] = eval_llm_judge(model, tokenizer, args.judge_n)

    import datetime
    results = {
        "model":      args.base_model,
        "adapter":    args.adapter_path,
        "timestamp":  datetime.datetime.now().isoformat(),
        "benchmarks": bench_results,
    }
    generate_report(results, Path(args.output_dir))
    logger.info("Evaluation complete → %s", args.output_dir)


if __name__ == "__main__":
    main()
