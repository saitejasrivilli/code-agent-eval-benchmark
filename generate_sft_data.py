"""
generate_sft_data.py

Takes eval_tool_use results and generates corrective SFT training data
for failed tasks. Each failure is converted to a correct demonstration
in ReAct format.

Usage:
    # Run eval first (or use existing results):
    python eval_tool_use.py --output results/tool_use_results.json

    # Generate SFT data from failures:
    python generate_sft_data.py \
        --input results/tool_use_results.json \
        --output data/sft_from_failures.jsonl \
        --augment 3

    # Mix in real API traces from Salesforce/xlam-function-calling-60k:
    python generate_sft_data.py \
        --input results/tool_use_results.json \
        --output data/sft_from_failures.jsonl \
        --real_data \
        --real_n 100
"""

from __future__ import annotations
import argparse
import json
import random
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval_tool_use import (
    TASKS,
    Task,
    TOOL_DESCRIPTIONS,
    eval_tool_use,
    _rule_based_agent,
)

# ── Correct ReAct traces keyed by task id ─────────────────────────────────────
# Each value is a list of (thought, tool_call_str, obs_template) tuples
# followed by the final_answer string.  obs_template uses {result} placeholder
# which is filled with the real tool output so the trace is self-consistent.

_SYSTEM_PROMPT = (
    "You are a precise reasoning agent. Solve the task step by step using tools.\n"
    + TOOL_DESCRIPTIONS
)

# Precomputed correct answers and minimal tool sequences for every task.
# Structured as list of assistant turn strings (may contain multiple turns
# before the final_answer is emitted).
_CORRECT_TRACES: dict[str, list[str]] = {
    # ── computation ──────────────────────────────────────────────────────────
    "fib_15": [
        textwrap.dedent("""\
            I need the 15th Fibonacci number. I'll compute it directly with Python.
            <tool_call>{"tool": "python_exec", "args": {"code": "a, b = 1, 1\\nfor _ in range(13):\\n    a, b = b, a + b\\nprint(b)"}}</tool_call>"""),
        "<final_answer>610</final_answer>",
    ],
    "primes_50": [
        textwrap.dedent("""\
            I'll count primes up to 50 using a sieve.
            <tool_call>{"tool": "python_exec", "args": {"code": "print(sum(1 for n in range(2, 51) if all(n % i for i in range(2, n))))"}}</tool_call>"""),
        "<final_answer>15</final_answer>",
    ],
    "sum_squares": [
        textwrap.dedent("""\
            Sum of squares 1..20 = sum(i^2).
            <tool_call>{"tool": "python_exec", "args": {"code": "print(sum(i * i for i in range(1, 21)))"}}</tool_call>"""),
        "<final_answer>2870</final_answer>",
    ],
    "factorial_10": [
        textwrap.dedent("""\
            10! = 1*2*...*10.
            <tool_call>{"tool": "calculator", "args": {"expr": "1*2*3*4*5*6*7*8*9*10"}}</tool_call>"""),
        "<final_answer>3628800</final_answer>",
    ],
    "compound_interest": [
        textwrap.dedent("""\
            Compound interest: A = P*(1+r)^n = 1000*(1.05)^10.
            <tool_call>{"tool": "calculator", "args": {"expr": "1000 * (1.05 ** 10)"}}</tool_call>"""),
        "<final_answer>1628.89</final_answer>",
    ],
    # ── multi_step ────────────────────────────────────────────────────────────
    "sqrt_then_check": [
        textwrap.dedent("""\
            Step 1: compute sqrt(144) + sqrt(169).
            <tool_call>{"tool": "calculator", "args": {"expr": "12 + 13"}}</tool_call>"""),
        textwrap.dedent("""\
            The result is 25. Step 2: check whether 25 is a perfect square.
            <tool_call>{"tool": "python_exec", "args": {"code": "import math; n=25; sq=int(math.isqrt(n)); print('yes' if sq*sq==n else 'no')"}}</tool_call>"""),
        "<final_answer>25 yes</final_answer>",
    ],
    "fizzbuzz_count": [
        textwrap.dedent("""\
            Count integers 1..100 divisible by 3 OR 5.
            <tool_call>{"tool": "python_exec", "args": {"code": "print(sum(1 for i in range(1, 101) if i % 3 == 0 or i % 5 == 0))"}}</tool_call>"""),
        "<final_answer>47</final_answer>",
    ],
    "digit_sum_factorial": [
        textwrap.dedent("""\
            Step 1: compute 10!.
            <tool_call>{"tool": "python_exec", "args": {"code": "import math; print(math.factorial(10))"}}</tool_call>"""),
        textwrap.dedent("""\
            Step 2: sum the digits of 3628800.
            <tool_call>{"tool": "python_exec", "args": {"code": "print(sum(int(d) for d in '3628800'))"}}</tool_call>"""),
        "<final_answer>27</final_answer>",
    ],
    "geometric_series": [
        textwrap.dedent("""\
            Geometric series 2 + 6 + ... + 2*3^8. Sum = a*(r^n - 1)/(r-1) = 2*(3^9 - 1)/2.
            <tool_call>{"tool": "calculator", "args": {"expr": "2 * (3**9 - 1) / (3 - 1)"}}</tool_call>"""),
        "<final_answer>19682</final_answer>",
    ],
    "collatz_steps": [
        textwrap.dedent("""\
            I'll simulate the Collatz sequence from 27 and count steps to reach 1.
            <tool_call>{"tool": "python_exec", "args": {"code": "n, s = 27, 0\\nwhile n != 1:\\n    n = n * 3 + 1 if n % 2 else n // 2\\n    s += 1\\nprint(s)"}}</tool_call>"""),
        "<final_answer>111</final_answer>",
    ],
    # ── knowledge_compute ─────────────────────────────────────────────────────
    "circle_area_pi": [
        textwrap.dedent("""\
            Step 1: look up the exact value of pi.
            <tool_call>{"tool": "lookup", "args": {"query": "pi"}}</tool_call>"""),
        textwrap.dedent("""\
            Pi = 3.141592653589793. Step 2: area = pi * r^2 = pi * 7^2.
            <tool_call>{"tool": "calculator", "args": {"expr": "3.141592653589793 * 7 * 7"}}</tool_call>"""),
        "<final_answer>153.9380</final_answer>",
    ],
    "light_travel": [
        textwrap.dedent("""\
            Step 1: look up the speed of light.
            <tool_call>{"tool": "lookup", "args": {"query": "speed of light"}}</tool_call>"""),
        textwrap.dedent("""\
            Speed of light = 299,792,458 m/s. In 1 microsecond = 1e-6 s: distance = 299792458 / 1000000.
            <tool_call>{"tool": "calculator", "args": {"expr": "299792458 / 1000000"}}</tool_call>"""),
        "<final_answer>299.792458</final_answer>",
    ],
    # ── error_recovery ────────────────────────────────────────────────────────
    "ambiguous_sqrt": [
        textwrap.dedent("""\
            I'll try sqrt(-1) first; if it errors I'll fall back to sqrt(1).
            <tool_call>{"tool": "python_exec", "args": {"code": "import math\\ntry:\\n    print(math.sqrt(-1))\\nexcept ValueError:\\n    print(math.sqrt(1))"}}</tool_call>"""),
        "<final_answer>1</final_answer>",
    ],
    "safe_division": [
        textwrap.dedent("""\
            I'll try 100/0; if it errors I'll compute 100/5 instead.
            <tool_call>{"tool": "python_exec", "args": {"code": "try:\\n    print(100 / 0)\\nexcept ZeroDivisionError:\\n    print(100 / 5)"}}</tool_call>"""),
        "<final_answer>20</final_answer>",
    ],
}

# ── Rephrasings for --augment ─────────────────────────────────────────────────

_REPHRASINGS: dict[str, list[str]] = {
    "fib_15": [
        "Find the 15th term in the Fibonacci sequence where the sequence starts 1, 1, 2, 3, ...",
        "In the Fibonacci series 1, 1, 2, 3, 5, ..., what is the value at position 15?",
        "Starting from F(1)=1, F(2)=1, what is F(15)?",
    ],
    "primes_50": [
        "Count all prime numbers that are less than or equal to 50.",
        "How many primes exist in the range [2, 50] inclusive?",
        "List primes up to 50 and state how many there are.",
    ],
    "sum_squares": [
        "Compute 1² + 2² + 3² + ... + 20².",
        "What is the sum of the squares of the first 20 positive integers?",
        "Calculate the sum of i² for i from 1 to 20.",
    ],
    "factorial_10": [
        "Evaluate 10! (ten factorial).",
        "What is the product of all positive integers from 1 to 10?",
        "Compute the factorial of the number ten.",
    ],
    "compound_interest": [
        "An investment of $1000 earns 5% annual compound interest. What is it worth after 10 years? Round to 2 decimals.",
        "What does $1000 grow to if compounded at 5% per year for 10 years? Give 2 decimal places.",
        "Calculate A = 1000*(1.05)^10 and round to 2 decimal places.",
    ],
    "sqrt_then_check": [
        "Add sqrt(144) and sqrt(169). Is their sum a perfect square? Give both the sum and yes/no.",
        "Compute the square root of 144 plus the square root of 169. Is this total a perfect square?",
        "What is √144 + √169? Is the answer a perfect square? Answer as '<number> yes/no'.",
    ],
    "fizzbuzz_count": [
        "Among integers 1 through 100, how many are divisible by 3 or by 5?",
        "How many numbers in [1, 100] satisfy: divisible by 3, or divisible by 5, or both?",
        "Count the FizzBuzz numbers from 1 to 100 (those divisible by 3 or 5).",
    ],
    "digit_sum_factorial": [
        "Sum the individual digits of 10 factorial.",
        "Compute 10! then add all its decimal digits together.",
        "What is the digit sum of the number 3628800?",
    ],
    "geometric_series": [
        "Find the total of the series 2, 6, 18, 54, ..., 2·3^8.",
        "Sum the geometric progression with first term 2, common ratio 3, and last term 2·3^8.",
        "What is 2 + 6 + 18 + 54 + 162 + 486 + 1458 + 4374 + 13122?",
    ],
    "collatz_steps": [
        "Starting at 27, apply the Collatz rule (odd → 3n+1, even → n/2) until reaching 1. How many applications does it take?",
        "How many iterations of the Collatz function are needed to get from 27 down to 1?",
        "Apply 3n+1 (for odd n) or n/2 (for even n) starting from 27. Count steps to reach 1.",
    ],
    "circle_area_pi": [
        "What is the area of a circle whose radius is 7 units? Use the precise mathematical value of pi and round to 4 decimal places.",
        "Compute π × 7² using the exact value of π, rounded to 4 decimals.",
        "A circle has radius 7. Calculate its area to 4 decimal places using the true value of π.",
    ],
    "light_travel": [
        "Given the exact speed of light, how many metres does light cover in one microsecond?",
        "Light travels at its exact speed. How far (in metres) does it go in 1 microsecond (10^-6 s)?",
        "Using the precise speed of light value, calculate the distance light travels in 0.000001 seconds.",
    ],
    "ambiguous_sqrt": [
        "Try to compute the square root of negative one. If that fails, return sqrt(1) instead.",
        "Attempt sqrt(-1). On error, fall back and compute sqrt(1).",
        "What is the square root of -1? If an error occurs, compute the square root of 1 as fallback.",
    ],
    "safe_division": [
        "Try dividing 100 by 0. If a division error occurs, divide 100 by 5 instead.",
        "Compute 100 ÷ 0. If that raises an error, compute 100 ÷ 5.",
        "Attempt the calculation 100/0; on failure, calculate 100/5.",
    ],
}


def _build_assistant_turn(turns: list[str]) -> str:
    """Join multi-turn assistant messages into a single training target string."""
    return "\n".join(turns)


# ── Real API trace helpers (Salesforce/xlam-function-calling-60k) ─────────────

def _format_tool_for_sft(tool: dict) -> str:
    """Format a single tool dict as a compact human-readable block."""
    name = tool.get("name", "unknown")
    desc = tool.get("description", "No description.")
    params = tool.get("parameters", {})
    props = params.get("properties", {}) if isinstance(params, dict) else {}
    required = params.get("required", []) if isinstance(params, dict) else []

    lines = [f"  {name}: {desc}"]
    if props:
        lines.append("    Parameters:")
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "any") if isinstance(pinfo, dict) else "any"
            pdesc = pinfo.get("description", "") if isinstance(pinfo, dict) else ""
            req_marker = " (required)" if pname in required else ""
            lines.append(f"      - {pname} ({ptype}){req_marker}: {pdesc}")
    return "\n".join(lines)


def _build_real_system_prompt(tools: list[dict]) -> str:
    tool_block = "\n\n".join(_format_tool_for_sft(t) for t in tools)
    return (
        "You are a function-calling assistant. You have access to these functions:\n\n"
        + tool_block
        + "\n\nWhen the user asks something, call the appropriate function. Respond with ONLY:\n"
        "<function_call>{\"name\": \"...\", \"arguments\": {...}}</function_call>"
    )


def _xlam_to_sft_example(ex: dict) -> dict[str, Any]:
    """
    Convert one xlam example to the SFT messages format.

    ex: {"query": str, "tools": list[dict], "answers": list[dict], "category": str}
    """
    tools: list[dict] = ex["tools"]
    answers: list[dict] = ex["answers"]
    first_call = answers[0]

    system_content = _build_real_system_prompt(tools)
    assistant_content = (
        "<function_call>"
        + json.dumps({"name": first_call.get("name", ""), "arguments": first_call.get("arguments", {})})
        + "</function_call>"
    )

    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": ex["query"]},
            {"role": "assistant", "content": assistant_content},
        ],
        "source": "real_api",
        "category": ex.get("category", "other"),
    }


def load_real_examples(n: int = 100, seed: int = 42) -> list[dict[str, Any]]:
    """
    Pull n examples from Salesforce/xlam-function-calling-60k and convert
    them to the SFT messages format.  Returns a list of example dicts.
    """
    from datasets import load_dataset

    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")

    raw: list[dict] = []
    for ex in ds:
        try:
            tools = json.loads(ex["tools"]) if isinstance(ex["tools"], str) else ex["tools"]
            answers = json.loads(ex["answers"]) if isinstance(ex["answers"], str) else ex["answers"]
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(tools, list) or not tools:
            continue
        if not isinstance(answers, list) or not answers:
            continue
        raw.append({
            "query": ex["query"],
            "tools": tools,
            "answers": answers,
            "category": ex.get("category") or ex.get("tool_category") or "other",
        })

    rng = random.Random(seed)
    rng.shuffle(raw)
    selected = raw[:n]

    return [_xlam_to_sft_example(ex) for ex in selected]


def _make_example(task: Task, description_override: str | None = None) -> dict[str, Any]:
    turns = _CORRECT_TRACES[task.id]
    assistant_content = _build_assistant_turn(turns)
    user_description = description_override or task.description
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Task: {user_description}"},
            {"role": "assistant", "content": assistant_content},
        ],
        "task_id": task.id,
        "category": task.category,
        "expected": task.expected,
    }


def _load_or_run_eval(input_path: str) -> dict:
    p = Path(input_path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    print(f"[info] {input_path} not found — running baseline eval with rule-based agent...")
    p.parent.mkdir(parents=True, exist_ok=True)
    return eval_tool_use(model_fn=_rule_based_agent, output_path=input_path)


def generate(
    input_path: str = "results/tool_use_results.json",
    output_path: str = "data/sft_from_failures.jsonl",
    augment: int = 1,
    real_data: bool = False,
    real_n: int = 100,
) -> dict[str, int]:
    """
    Main generation routine.

    Returns a stats dict: {category: n_examples, "skipped": n_skipped,
    optionally "real_api": n_real}.
    """
    eval_data = _load_or_run_eval(input_path)

    # Index failed task ids from the eval results
    failed_ids: set[str] = {
        t["id"] for t in eval_data["tasks"] if not t["success"]
    }
    # Index tasks by id for O(1) lookup
    task_by_id: dict[str, Task] = {t.id: t for t in TASKS}

    stats: dict[str, int] = {}
    n_skipped = 0
    examples: list[dict] = []

    for task_record in eval_data["tasks"]:
        tid = task_record["id"]
        if tid not in failed_ids:
            n_skipped += 1
            continue
        if tid not in task_by_id:
            print(f"[warn] task id '{tid}' not found in TASKS — skipping")
            continue
        if tid not in _CORRECT_TRACES:
            print(f"[warn] no correct trace for '{tid}' — skipping")
            continue

        task = task_by_id[tid]
        category = task.category
        stats[category] = stats.get(category, 0)

        # Base example
        examples.append(_make_example(task))
        stats[category] += 1

        # Augmented variants
        rephrasings = _REPHRASINGS.get(tid, [])
        for i in range(1, augment):
            if i - 1 < len(rephrasings):
                alt_desc = rephrasings[i - 1]
            else:
                # Fall back to appending a mild prefix variation
                prefixes = [
                    "Please solve the following: ",
                    "Using appropriate tools, answer: ",
                    "Determine the answer to: ",
                ]
                alt_desc = prefixes[(i - 1) % len(prefixes)] + task.description
            examples.append(_make_example(task, description_override=alt_desc))
            stats[category] += 1

    # ── Optionally mix in real API traces ─────────────────────────────────────
    if real_data and real_n > 0:
        print(f"[info] Loading {real_n} real examples from Salesforce/xlam-function-calling-60k…")
        real_examples = load_real_examples(n=real_n)
        examples.extend(real_examples)

        cat_counts: dict[str, int] = defaultdict(int)
        for ex in real_examples:
            cat_counts[ex.get("category", "other")] += 1

        top_cats = sorted(cat_counts.items(), key=lambda x: -x[1])[:8]
        cat_str = ", ".join(f"{c}={n}" for c, n in top_cats)
        print(
            f"Added {len(real_examples)} real examples from "
            f"Salesforce/xlam-function-calling-60k (categories: {cat_str})"
        )
        stats["real_api"] = len(real_examples)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    stats["skipped_successful"] = n_skipped
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate corrective SFT data from eval_tool_use failures."
    )
    parser.add_argument(
        "--input",
        default="results/tool_use_results.json",
        help="Path to tool_use_results.json produced by eval_tool_use.py",
    )
    parser.add_argument(
        "--output",
        default="data/sft_from_failures.jsonl",
        help="Output JSONL path for SFT training examples",
    )
    parser.add_argument(
        "--augment",
        type=int,
        default=1,
        metavar="N",
        help="Number of examples per failed task (1 = no augmentation, 3 = original + 2 rephrasings)",
    )
    parser.add_argument(
        "--real_data",
        action="store_true",
        help="Also pull real API traces from Salesforce/xlam-function-calling-60k",
    )
    parser.add_argument(
        "--real_n",
        type=int,
        default=100,
        metavar="N",
        help="Number of real API examples to add (default 100, requires --real_data)",
    )
    args = parser.parse_args()

    print(f"Input : {args.input}")
    print(f"Output: {args.output}")
    print(f"Augment factor: {args.augment}")
    if args.real_data:
        print(f"Real API examples: {args.real_n} (Salesforce/xlam-function-calling-60k)")
    print()

    stats = generate(
        input_path=args.input,
        output_path=args.output,
        augment=args.augment,
        real_data=args.real_data,
        real_n=args.real_n,
    )

    n_skipped = stats.pop("skipped_successful", 0)
    n_real = stats.pop("real_api", 0)
    total = sum(stats.values()) + n_real

    print("Examples generated per category (synthetic failures):")
    for cat, n in sorted(stats.items()):
        print(f"  {cat:22s}  {n}")
    if n_real:
        print(f"  {'real_api':22s}  {n_real}")
    print(f"  {'TOTAL':22s}  {total}")
    print(f"\nSkipped (already successful): {n_skipped}")
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
