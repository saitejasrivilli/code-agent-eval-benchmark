"""
Multi-agent pipeline: Planner → Executor → Critic

Motivation
----------
The single ReAct agent in eval_tool_use.py treats every task the same way:
think, call a tool, observe, repeat. On multi-step tasks this leads to two
failure modes: (1) the agent gets lost mid-sequence and hits the step limit,
(2) a correct-looking but numerically wrong answer goes unchallenged.

A three-agent pipeline fixes both:
  Planner  — decomposes the task into an explicit ordered plan before any
             tool calls are made, giving the Executor a roadmap.
  Executor — runs the existing ReAct loop but anchored to the plan, so it
             knows which subgoal it is currently working on.
  Critic   — after the Executor produces an answer, independently verifies
             it; if the answer is wrong it produces corrective feedback and
             the Executor gets one retry.

Usage
-----
  python multi_agent_pipeline.py --mode demo       # deterministic, no GPU
  python multi_agent_pipeline.py --mode hf \
      --model Qwen/Qwen2.5-7B-Instruct             # real model
  python multi_agent_pipeline.py --compare         # single vs multi-agent
"""

from __future__ import annotations
import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Callable

from eval_tool_use import (
    TASKS, Task, TaskResult, StepRecord,
    TOOLS, TOOL_DESCRIPTIONS,
    _answers_match, _TOOL_CALL_RE, _FINAL_RE, _parse_tool_call,
    _rule_based_agent, eval_tool_use,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    steps: list[str]

    def as_context(self) -> str:
        lines = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(self.steps))
        return f"Your plan:\n{lines}\n\nExecute each step in order."


@dataclass
class CritiqueResult:
    approved: bool
    feedback: str


@dataclass
class MultiAgentResult:
    task_id: str
    category: str
    success: bool
    final_answer: str
    expected: str
    plan: Plan
    critique: CritiqueResult
    retried: bool
    latency_s: float


# ---------------------------------------------------------------------------
# Planner agent
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = (
    "You are a planning agent. Given a task, output a numbered list of "
    "concrete steps to solve it. Each step should be one clear action. "
    "Do not solve the task — only plan. Output ONLY the numbered list.\n"
    "Example:\n1. Calculate X\n2. Look up Y\n3. Combine results"
)

_CATEGORY_PLANS: dict[str, list[str]] = {
    "computation": [
        "Identify the computation required.",
        "Write Python code to compute the answer.",
        "Execute the code and record the result.",
    ],
    "multi_step": [
        "Break the problem into sequential sub-computations.",
        "Execute each sub-computation with python_exec.",
        "Combine intermediate results into the final answer.",
    ],
    "knowledge_compute": [
        "Look up the required constant from the knowledge base.",
        "Identify the formula to apply.",
        "Compute the result using the constant.",
    ],
    "error_recovery": [
        "Attempt the primary computation.",
        "If the tool returns an error, identify the invalid input.",
        "Retry with the corrected input.",
    ],
}


def plan(task: Task, model_fn: Callable[[str], str]) -> Plan:
    """Generate a plan for the task. model_fn is called once."""
    prompt = f"{_PLANNER_SYSTEM}\n\nTask: {task.description}"
    raw = model_fn(prompt)
    steps = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if line and line[0].isdigit():
            # strip leading "1. " or "1) "
            text = line.split(".", 1)[-1].split(")", 1)[-1].strip()
            if text:
                steps.append(text)
    return Plan(steps=steps or ["Solve the task step by step using tools."])


def _rule_based_planner(task: Task) -> Plan:
    steps = _CATEGORY_PLANS.get(task.category, _CATEGORY_PLANS["computation"])
    return Plan(steps=list(steps))


# ---------------------------------------------------------------------------
# Executor agent — extends ReAct with plan as prefix context
# ---------------------------------------------------------------------------

def execute(task: Task, plan: Plan, model_fn: Callable[[str], str]) -> TaskResult:
    """
    Run the ReAct loop with the plan prepended as system context.
    Reuses all the existing tool infrastructure from eval_tool_use.
    """
    t0 = time.time()
    system = (
        "You are a precise reasoning agent. Solve the task step by step using tools.\n"
        + TOOL_DESCRIPTIONS + "\n\n"
        + plan.as_context()
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
        context = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        response = model_fn(context)
        messages.append({"role": "assistant", "content": response})

        fa_match = _FINAL_RE.search(response)
        if fa_match:
            final_answer = fa_match.group(1).strip()
            success = _answers_match(final_answer, task.expected)
            break

        tool_matches = _TOOL_CALL_RE.findall(response)
        if not tool_matches:
            break

        for raw in tool_matches:
            parsed = _parse_tool_call(raw)
            if parsed is None:
                result_text, is_error = "ERROR: malformed tool call JSON", True
            else:
                tool_name, args = parsed
                fn = TOOLS.get(tool_name)
                if fn is None:
                    result_text, is_error = f"ERROR: unknown tool '{tool_name}'", True
                else:
                    result_text = fn(**args)
                    is_error = result_text.startswith("ERROR")

            if is_error:
                had_error = True
            elif had_error:
                error_recovered = True

            steps.append(StepRecord(
                step=step_num,
                tool=parsed[0] if parsed else "unknown",
                args=parsed[1] if parsed else {},
                result=result_text,
                is_error=is_error,
            ))
            messages.append({"role": "tool", "content": f"[OBS] {result_text}"})

    return TaskResult(
        task_id=task.id, category=task.category,
        success=success, final_answer=final_answer, expected=task.expected,
        steps=steps, error_recovered=error_recovered,
        latency_s=round(time.time() - t0, 3),
        hit_step_limit=(step_num >= task.max_steps and not success),
    )


# ---------------------------------------------------------------------------
# Critic agent
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM = (
    "You are a verification agent. Given a task and a proposed answer, "
    "determine whether the answer is correct.\n"
    "Respond with exactly:\n"
    "  APPROVED: <brief reason>\n"
    "or:\n"
    "  REJECTED: <specific corrective feedback for the solver>"
)


def critique(
    task: Task,
    answer: str,
    model_fn: Callable[[str], str],
) -> CritiqueResult:
    """Ask the Critic to verify the Executor's answer."""
    prompt = (
        f"{_CRITIC_SYSTEM}\n\n"
        f"Task: {task.description}\n"
        f"Proposed answer: {answer}"
    )
    raw = model_fn(prompt).strip()
    if raw.upper().startswith("APPROVED"):
        return CritiqueResult(approved=True, feedback=raw)
    return CritiqueResult(
        approved=False,
        feedback=raw.replace("REJECTED:", "").strip() or "The answer appears incorrect.",
    )


def _rule_based_critic(task: Task, answer: str) -> CritiqueResult:
    """Deterministic critic for demo mode — checks numeric equality."""
    if _answers_match(answer, task.expected):
        return CritiqueResult(approved=True, feedback="Answer matches expected value.")
    return CritiqueResult(
        approved=False,
        feedback=(
            f"The answer '{answer}' does not appear correct. "
            f"Re-examine your computation — check for off-by-one errors "
            f"or incorrect formula application."
        ),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_multi_agent(
    task: Task,
    model_fn: Callable[[str], str],
    demo: bool = False,
) -> MultiAgentResult:
    """
    Full Planner → Executor → Critic pipeline with one retry on rejection.

    Parameters
    ----------
    demo : bool
        If True, uses deterministic rule-based agents (no LLM calls needed).
        Useful for CI and benchmarking without a GPU.
    """
    t0 = time.time()

    # 1. Plan
    task_plan = _rule_based_planner(task) if demo else plan(task, model_fn)

    # 2. Execute
    exec_model = _rule_based_agent if demo else model_fn
    result = execute(task, task_plan, exec_model)

    # 3. Critique
    crit = (
        _rule_based_critic(task, result.final_answer)
        if demo
        else critique(task, result.final_answer, model_fn)
    )

    retried = False

    # 4. Retry once with critic feedback injected as context
    if not crit.approved:
        def feedback_model_fn(prompt: str) -> str:
            augmented = prompt + f"\n\nCRITIC FEEDBACK: {crit.feedback}\nPlease correct your answer."
            return exec_model(augmented)

        retry_result = execute(task, task_plan, feedback_model_fn)
        retried = True
        if retry_result.success or (not result.success):
            result = retry_result

    return MultiAgentResult(
        task_id=task.id,
        category=task.category,
        success=result.success,
        final_answer=result.final_answer,
        expected=task.expected,
        plan=task_plan,
        critique=crit,
        retried=retried,
        latency_s=round(time.time() - t0, 3),
    )


def eval_multi_agent(tasks: list[Task], demo: bool = False) -> dict:
    results = [run_multi_agent(t, _rule_based_agent, demo=demo) for t in tasks]
    n = len(results)
    return {
        "task_success":    round(sum(r.success for r in results) / n, 3),
        "retry_rate":      round(sum(r.retried for r in results) / n, 3),
        "critic_approval": round(sum(r.critique.approved for r in results) / n, 3),
        "n_tasks":         n,
        "results":         results,
    }


# ---------------------------------------------------------------------------
# Comparison: single-agent vs multi-agent
# ---------------------------------------------------------------------------

def compare(demo: bool = True) -> None:
    print("Single-agent (ReAct only):")
    single = eval_tool_use(TASKS, _rule_based_agent)
    print(f"  task_success={single['task_success']}  "
          f"tool_accuracy={single['tool_accuracy']}  "
          f"avg_steps={single['avg_steps']}")

    print("\nMulti-agent (Planner → Executor → Critic):")
    multi = eval_multi_agent(TASKS, demo=demo)
    print(f"  task_success={multi['task_success']}  "
          f"retry_rate={multi['retry_rate']}  "
          f"critic_approval={multi['critic_approval']}")

    print("\nDelta:")
    delta = round(multi["task_success"] - single["task_success"], 3)
    print(f"  task_success delta={delta:+.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demo", "hf"], default="demo")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    if args.compare:
        compare(demo=(args.mode == "demo"))
    else:
        demo = args.mode == "demo"
        results = eval_multi_agent(TASKS, demo=demo)
        print(json.dumps({k: v for k, v in results.items() if k != "results"}, indent=2))
