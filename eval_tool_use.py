"""
Multi-step tool-use agent evaluation.

Evaluates a language model as a ReAct agent that can call tools across
multiple turns to solve tasks requiring sequential reasoning.

Tools available to the agent:
  python_exec  — execute Python code and return stdout/result
  calculator   — evaluate a math expression safely
  lookup       — retrieve a fact from a small hardcoded knowledge base

Metrics:
  task_success     — fraction of tasks solved correctly (verifiable answer)
  avg_steps        — mean tool calls per task
  tool_accuracy    — fraction of tool calls that return a non-error result
  error_recovery   — fraction of tasks where agent recovered after a tool error
  avg_latency_s    — wall-clock seconds per task
"""

from __future__ import annotations
import ast
import json
import math
import re
import subprocess
import sys
import time
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Tool implementations ──────────────────────────────────────────────────────

def tool_python_exec(code: str, timeout: int = 5) -> str:
    """Execute Python code in a subprocess sandbox, return stdout or error."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if result.returncode != 0:
            return f"ERROR: {err[:300]}"
        return out[:500] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: execution timed out"
    except Exception as e:
        return f"ERROR: {e}"


def tool_calculator(expr: str) -> str:
    """Safely evaluate a math expression (no arbitrary code)."""
    allowed = set("0123456789+-*/().% eE")
    clean = expr.strip()
    if not all(c in allowed for c in clean):
        return f"ERROR: unsafe expression '{clean}'"
    try:
        result = eval(clean, {"__builtins__": {}, "math": math})  # noqa: S307
        return str(round(result, 8)) if isinstance(result, float) else str(result)
    except Exception as e:
        return f"ERROR: {e}"


_KNOWLEDGE_BASE = {
    "speed of light": "299,792,458 m/s",
    "pi": "3.141592653589793",
    "avogadro": "6.02214076e23",
    "planck": "6.62607015e-34 J·s",
    "euler number": "2.718281828459045",
    "boiling point of water": "100°C / 212°F at 1 atm",
    "freezing point of water": "0°C / 32°F",
    "gravitational constant": "6.674e-11 N·m²/kg²",
}

def tool_lookup(query: str) -> str:
    q = query.lower().strip()
    for key, val in _KNOWLEDGE_BASE.items():
        if key in q or q in key:
            return val
    return f"NOT_FOUND: '{query}'"


TOOLS = {
    "python_exec": tool_python_exec,
    "calculator": tool_calculator,
    "lookup": tool_lookup,
}

TOOL_DESCRIPTIONS = """You have access to these tools. Call them using the exact format shown.

<tool_call>{"tool": "python_exec", "args": {"code": "print(sum(range(10)))"}}</tool_call>
<tool_call>{"tool": "calculator", "args": {"expr": "2 ** 10 + 3 * 7"}}</tool_call>
<tool_call>{"tool": "lookup", "args": {"query": "speed of light"}}</tool_call>

After each tool call you will receive an [OBS] tag with the result.
When you have the final answer, output it as: <final_answer>VALUE</final_answer>
"""

_OBS_MARKER = "[OBS]"

# ── Task definitions ──────────────────────────────────────────────────────────

@dataclass
class Task:
    id: str
    description: str
    expected: str          # canonical answer (string match after normalisation)
    max_steps: int = 6
    category: str = "general"


TASKS: list[Task] = [
    # Pure computation (requires python_exec or calculator)
    Task("fib_15", "What is the 15th Fibonacci number (1-indexed, starting 1,1,2,...)?",
         "610", category="computation"),
    Task("primes_50", "How many prime numbers are there up to and including 50?",
         "15", category="computation"),
    Task("sum_squares", "What is the sum of squares of integers from 1 to 20?",
         "2870", category="computation"),
    Task("factorial_10", "What is 10 factorial?",
         "3628800", category="computation"),
    Task("compound_interest",
         "What is $1000 compounded annually at 5% for 10 years? Round to 2 decimal places.",
         "1628.89", category="computation"),

    # Multi-step reasoning (requires multiple tool calls)
    Task("sqrt_then_check",
         "Compute sqrt(144) + sqrt(169). Is the result a perfect square?",
         "25 yes", category="multi_step"),
    Task("fizzbuzz_count",
         "How many integers from 1 to 100 are divisible by 3 OR by 5 (FizzBuzz numbers)?",
         "47", category="multi_step"),
    Task("digit_sum_factorial",
         "What is the digit sum of 10 factorial?",
         "27", category="multi_step"),
    Task("geometric_series",
         "What is the sum of the geometric series 2 + 6 + 18 + ... + 2*3^8?",
         "19682", category="multi_step"),
    Task("collatz_steps",
         "How many steps does the Collatz sequence take to reach 1 starting from 27?",
         "111", category="multi_step"),

    # Knowledge + computation (requires lookup then calculator)
    Task("circle_area_pi",
         "Using the exact value of pi, what is the area of a circle with radius 7? Round to 4 decimal places.",
         "153.9380", category="knowledge_compute"),
    Task("light_travel",
         "How far does light travel in 1 microsecond (in metres)? Use the exact speed of light.",
         "299.792458", category="knowledge_compute"),

    # Error recovery (deliberately ambiguous first approach)
    Task("ambiguous_sqrt",
         "What is sqrt(-1)? If that fails, compute sqrt(1) instead.",
         "1", category="error_recovery"),
    Task("safe_division",
         "Compute 100 / 0. If that raises an error, compute 100 / 5 instead.",
         "20", category="error_recovery"),
]

# ── ReAct agent loop ──────────────────────────────────────────────────────────

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)
_FINAL_RE = re.compile(r"<final_answer>(.*?)(?:</final_answer>|$)", re.DOTALL | re.IGNORECASE)


def _parse_tool_call(raw: str) -> tuple[str, dict] | None:
    raw = raw.strip()
    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        try:
            import ast
            obj = ast.literal_eval(raw)
        except Exception:
            pass
    if not isinstance(obj, dict) or "tool" not in obj:
        return None
    return obj["tool"], obj.get("args", {})


def _normalise_answer(text: str) -> str:
    text = text.lower().strip()
    # Extract numeric value if present
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return " ".join(nums)
    return text


def _answers_match(predicted: str, expected: str) -> bool:
    p = _normalise_answer(predicted)
    e = _normalise_answer(expected)
    if p == e:
        return True
    # Numeric tolerance
    p_nums = re.findall(r"-?\d+(?:\.\d+)?", p)
    e_nums = re.findall(r"-?\d+(?:\.\d+)?", e)
    if p_nums and e_nums:
        try:
            return all(
                abs(float(pn) - float(en)) < 0.01
                for pn, en in zip(p_nums, e_nums)
            )
        except ValueError:
            pass
    return False


@dataclass
class StepRecord:
    step: int
    tool: str
    args: dict
    result: str
    is_error: bool


@dataclass
class TaskResult:
    task_id: str
    category: str
    success: bool
    final_answer: str
    expected: str
    steps: list[StepRecord] = field(default_factory=list)
    error_recovered: bool = False
    latency_s: float = 0.0
    hit_step_limit: bool = False


def run_react_agent(
    task: Task,
    model_fn=None,     # callable(prompt: str) -> str  — raw concat format
    react_fn=None,     # callable(messages: list[dict]) -> str  — chat template format
) -> TaskResult:
    """
    Run a single ReAct loop.

    Prefer react_fn (apply_chat_template) for instruct-tuned models.
    model_fn (raw concatenated string) is kept for backwards compatibility
    and rule-based agents.
    """
    t0 = time.time()
    system = (
        "You are a precise reasoning agent. Solve the task step by step using tools.\n"
        + TOOL_DESCRIPTIONS
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Task: {task.description}"},
    ]

    steps: list[StepRecord] = []
    had_error = False
    error_recovered = False
    final_answer = ""
    success = False

    for step_num in range(1, task.max_steps + 1):
        if react_fn is not None:
            response = react_fn(messages)
        else:
            context = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages
            )
            response = model_fn(context)
        messages.append({"role": "assistant", "content": response})

        # Check for final answer first
        fa_match = _FINAL_RE.search(response)
        if fa_match:
            final_answer = fa_match.group(1).strip()
            success = _answers_match(final_answer, task.expected)
            break

        # Parse tool calls
        tool_matches = _TOOL_CALL_RE.findall(response)
        if not tool_matches:
            # Model responded without a tool call or final answer — treat as stuck
            break

        for raw in tool_matches:
            parsed = _parse_tool_call(raw)
            if parsed is None:
                result_text = "ERROR: malformed tool call JSON"
                is_error = True
            else:
                tool_name, args = parsed
                fn = TOOLS.get(tool_name)
                if fn is None:
                    result_text = f"ERROR: unknown tool '{tool_name}'"
                    is_error = True
                else:
                    result_text = fn(**args)
                    is_error = result_text.startswith("ERROR")

            if is_error:
                had_error = True
            elif had_error:
                error_recovered = True

            steps.append(StepRecord(
                step=step_num, tool=parsed[0] if parsed else "unknown",
                args=parsed[1] if parsed else {}, result=result_text,
                is_error=is_error,
            ))
            messages.append({
                "role": "tool",
                "content": f"[OBS] {result_text}",
            })

    hit_limit = step_num >= task.max_steps and not success

    return TaskResult(
        task_id=task.id,
        category=task.category,
        success=success,
        final_answer=final_answer,
        expected=task.expected,
        steps=steps,
        error_recovered=error_recovered,
        latency_s=round(time.time() - t0, 3),
        hit_step_limit=hit_limit,
    )


# ── Evaluation harness ────────────────────────────────────────────────────────

def build_model_fn(model, tokenizer, max_new_tokens: int = 256):
    """Wrap a HuggingFace model into a callable(prompt) -> str."""
    import torch
    def _call(prompt: str) -> str:
        enc = tokenizer(
            prompt[-3000:],  # truncate long contexts
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(next(model.parameters()).device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=1.0,
            )
        generated = out[0][enc["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)
    return _call


def build_react_fn(model, tokenizer, max_new_tokens: int = 512):
    """
    Multi-turn ReAct fn that uses apply_chat_template.

    Instruct-tuned models (Qwen, Llama-3-Instruct, etc.) require the chat
    template so the model knows to generate as an assistant. Without it they
    continue the concatenated string as plain text and never emit tool calls.

    The 'tool' role is remapped to 'user' (observation wrapped in [OBS])
    because most models' chat templates only accept system/user/assistant.
    Consecutive user/tool messages are merged to avoid template validation
    errors on double-user-turn sequences.
    """
    import torch
    ROLE_MAP = {"system": "system", "user": "user", "assistant": "assistant", "tool": "user"}

    def fn(messages: list[dict]) -> str:
        normalized: list[dict] = []
        for m in messages:
            role = ROLE_MAP.get(m["role"], "user")
            if normalized and normalized[-1]["role"] == "user" and role == "user":
                normalized[-1] = {"role": "user",
                                   "content": normalized[-1]["content"] + "\n" + m["content"]}
            else:
                normalized.append({"role": role, "content": m["content"]})

        text = tokenizer.apply_chat_template(
            normalized, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(
            next(model.parameters()).device
        )
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = out[0][enc["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)
    return fn


def eval_tool_use(
    model=None,
    tokenizer=None,
    model_fn=None,
    react_fn=None,
    tasks: list[Task] | None = None,
    output_path: str | None = None,
) -> dict:
    """
    Run full tool-use evaluation.

    Pass either (model, tokenizer) for a HuggingFace model, or model_fn
    directly for a custom callable.
    """
    assert (model_fn is not None or react_fn is not None
            or (model is not None and tokenizer is not None)), \
        "provide model+tokenizer, model_fn, or react_fn"

    if react_fn is None and model_fn is None:
        react_fn = build_react_fn(model, tokenizer)
    if tasks is None:
        tasks = TASKS

    results: list[TaskResult] = []
    for task in tasks:
        r = run_react_agent(task, model_fn=model_fn, react_fn=react_fn)
        results.append(r)
        status = "✓" if r.success else "✗"
        print(f"  [{status}] {task.id:30s} steps={len(r.steps)} answer={r.final_answer!r}")

    # Aggregate metrics
    n = len(results)
    task_success     = sum(r.success for r in results) / n
    avg_steps        = sum(len(r.steps) for r in results) / n
    all_steps        = [s for r in results for s in r.steps]
    tool_accuracy    = sum(1 for s in all_steps if not s.is_error) / max(len(all_steps), 1)
    error_tasks      = [r for r in results if any(s.is_error for s in r.steps)]
    error_recovery   = (
        sum(r.error_recovered for r in error_tasks) / len(error_tasks)
        if error_tasks else 1.0
    )
    avg_latency      = sum(r.latency_s for r in results) / n

    # Per-category
    categories: dict[str, dict] = {}
    for r in results:
        cat = r.category
        if cat not in categories:
            categories[cat] = {"total": 0, "success": 0}
        categories[cat]["total"] += 1
        categories[cat]["success"] += r.success
    for cat in categories:
        categories[cat]["accuracy"] = round(
            categories[cat]["success"] / categories[cat]["total"], 4
        )

    summary = {
        "n_tasks": n,
        "task_success": round(task_success, 4),
        "avg_steps_per_task": round(avg_steps, 2),
        "tool_accuracy": round(tool_accuracy, 4),
        "error_recovery_rate": round(error_recovery, 4),
        "avg_latency_s": round(avg_latency, 3),
        "per_category": categories,
        "tasks": [
            {
                "id": r.task_id,
                "category": r.category,
                "success": r.success,
                "final_answer": r.final_answer,
                "expected": r.expected,
                "n_steps": len(r.steps),
                "error_recovered": r.error_recovered,
                "hit_step_limit": r.hit_step_limit,
                "latency_s": r.latency_s,
            }
            for r in results
        ],
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

    return summary


# ── Standalone demo with rule-based agent (no LLM required) ──────────────────

def _rule_based_agent(prompt: str) -> str:
    """
    Deterministic agent that always issues the correct tool call for each task.
    Used to measure the harness itself and produce a 'ceiling' baseline.
    """
    p = prompt.lower()
    # Count actual tool results (injected as "TOOL: [OBS] ..."), not the description text
    obs_count = sum(1 for line in prompt.split("\n") if line.startswith("TOOL:"))

    # digit_sum must come before 10_factorial (both contain "factorial")
    if "digit sum" in p and "factorial" in p and obs_count == 0:
        return '<tool_call>{"tool": "python_exec", "args": {"code": "import math; print(sum(int(d) for d in str(math.factorial(10))))"}}</tool_call>'
    if "fibonacci" in p and "15" in p and obs_count == 0:
        return '<tool_call>{"tool": "python_exec", "args": {"code": "a,b=1,1\\nfor _ in range(14):\\n a,b=b,a+b\\nprint(a)"}}</tool_call>'
    if "prime" in p and "50" in p and obs_count == 0:
        return '<tool_call>{"tool": "python_exec", "args": {"code": "print(sum(1 for n in range(2,51) if all(n%i for i in range(2,n))))"}}</tool_call>'
    if "sum of squares" in p and obs_count == 0:
        return '<tool_call>{"tool": "python_exec", "args": {"code": "print(sum(i*i for i in range(1,21)))"}}</tool_call>'
    if "10 factorial" in p and obs_count == 0:
        return '<tool_call>{"tool": "calculator", "args": {"expr": "1*2*3*4*5*6*7*8*9*10"}}' + "</tool_call>"
    if "compound" in p and obs_count == 0:
        return '<tool_call>{"tool": "calculator", "args": {"expr": "1000 * (1.05 ** 10)"}}' + "</tool_call>"
    if "sqrt(144)" in p and obs_count == 0:
        return '<tool_call>{"tool": "calculator", "args": {"expr": "12 + 13"}}' + "</tool_call>"
    if "fizzbuzz" in p and obs_count == 0:
        return '<tool_call>{"tool": "python_exec", "args": {"code": "print(sum(1 for i in range(1,101) if i%3==0 or i%5==0))"}}</tool_call>'
    if "geometric" in p and obs_count == 0:
        return '<tool_call>{"tool": "calculator", "args": {"expr": "2 * (3**9 - 1) / (3 - 1)"}}' + "</tool_call>"
    if "collatz" in p and obs_count == 0:
        return '<tool_call>{"tool": "python_exec", "args": {"code": "n,s=27,0\\nwhile n!=1:\\n n=n*3+1 if n%2 else n//2\\n s+=1\\nprint(s)"}}</tool_call>'
    if "area of a circle" in p and obs_count == 0:
        return '<tool_call>{"tool": "lookup", "args": {"query": "pi"}}' + "</tool_call>"
    if "area of a circle" in p and "3.14159" in prompt and obs_count == 1:
        return '<tool_call>{"tool": "calculator", "args": {"expr": "3.141592653589793 * 7 * 7"}}' + "</tool_call>"
    if "light travel" in p and obs_count == 0:
        return '<tool_call>{"tool": "lookup", "args": {"query": "speed of light"}}' + "</tool_call>"
    if "light travel" in p and "299,792,458" in prompt and obs_count == 1:
        return '<tool_call>{"tool": "calculator", "args": {"expr": "299792458 / 1000000"}}' + "</tool_call>"
    # Use python_exec for sqrt(-1): math.sqrt(-1) raises ValueError → subprocess returns ERROR
    if "sqrt(-1)" in p and obs_count == 0:
        return '<tool_call>{"tool": "python_exec", "args": {"code": "import math; print(math.sqrt(-1))"}}</tool_call>'
    if "sqrt(-1)" in p and "ERROR" in prompt and obs_count == 1:
        return '<tool_call>{"tool": "calculator", "args": {"expr": "1 ** 0.5"}}' + "</tool_call>"
    if "100 / 0" in p and obs_count == 0:
        return '<tool_call>{"tool": "calculator", "args": {"expr": "100 / 0"}}' + "</tool_call>"
    if "100 / 0" in p and "ERROR" in prompt and obs_count == 1:
        return '<tool_call>{"tool": "calculator", "args": {"expr": "100 / 5"}}' + "</tool_call>"

    # Extract last [OBS] result (formatted as "TOOL: [OBS] <value>") and emit final answer
    obs_lines = [
        line.split("[OBS] ", 1)[1].strip()
        for line in prompt.split("\n")
        if "[OBS] " in line
    ]
    if obs_lines:
        last = obs_lines[-1]
        if not last.startswith("ERROR"):
            return f"<final_answer>{last}</final_answer>"
    return "<final_answer>unknown</final_answer>"


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["demo", "hf"], default="demo",
                   help="demo=rule-based ceiling; hf=HuggingFace model")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--output", default="results/tool_use_results.json")
    args = p.parse_args()

    print(f"=== Multi-step Tool-Use Eval ({len(TASKS)} tasks) ===\n")

    if args.mode == "demo":
        summary = eval_tool_use(model_fn=_rule_based_agent, output_path=args.output)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        tok = AutoTokenizer.from_pretrained(args.model)
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, device_map="auto"
        )
        rfn = build_react_fn(mdl, tok)
        summary = eval_tool_use(react_fn=rfn, output_path=args.output)

    print(f"\n=== Results ===")
    print(f"  Task success:      {summary['task_success']:.1%}  ({int(summary['task_success']*summary['n_tasks'])}/{summary['n_tasks']})")
    print(f"  Avg steps/task:    {summary['avg_steps_per_task']:.1f}")
    print(f"  Tool accuracy:     {summary['tool_accuracy']:.1%}")
    print(f"  Error recovery:    {summary['error_recovery_rate']:.1%}")
    print(f"  Avg latency:       {summary['avg_latency_s']:.3f}s")
    print(f"\nPer category:")
    for cat, m in summary["per_category"].items():
        print(f"  {cat:20s}  {m['accuracy']:.1%}  ({m['success']}/{m['total']})")
    if args.output:
        print(f"\nSaved → {args.output}")
