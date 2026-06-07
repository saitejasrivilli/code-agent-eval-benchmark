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
    _rule_based_agent, eval_tool_use, build_model_fn, build_react_fn,
)


def build_chat_fn(model, tokenizer, max_new_tokens: int = 256) -> Callable[[str], str]:
    """
    One-shot chat wrapper using apply_chat_template.

    Used for Planner and Critic agents which send a single structured
    prompt (not a multi-turn ReAct trace). Without this, instruct-tuned
    models don't receive their expected <|im_start|> formatting and
    produce garbage instead of APPROVED/REJECTED.

    The Executor continues using the raw build_model_fn because its
    context is already a multi-turn ReAct trace, not a chat message.
    """
    def fn(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(
            model.device
        )
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = out[0][enc["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)
    return fn


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

def execute(
    task: Task,
    plan: Plan,
    model_fn: Callable[[str], str] | None = None,
    react_fn: Callable[[list], str] | None = None,
) -> TaskResult:
    """
    Run the ReAct loop with the plan prepended as system context.

    react_fn (apply_chat_template) is preferred for instruct-tuned models.
    model_fn (raw concat string) is kept for rule-based / demo mode.
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
        if react_fn is not None:
            response = react_fn(messages)
        else:
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
    model_fn: Callable[[str], str] | None = None,
    chat_fn: Callable[[str], str] | None = None,
    react_fn: Callable[[list], str] | None = None,
    demo: bool = False,
) -> MultiAgentResult:
    """
    Full Planner → Executor → Critic pipeline with one retry on rejection.

    Parameters
    ----------
    react_fn : callable(messages: list[dict]) -> str
        Chat-template-aware fn for the Executor's multi-turn ReAct loop.
        Required for instruct-tuned models; overrides model_fn for Executor.
    chat_fn : callable(prompt: str) -> str
        apply_chat_template one-shot fn for Planner + Critic.
    model_fn : callable(prompt: str) -> str
        Raw concat fn — used for demo/rule-based mode or as fallback.
    demo : bool
        If True, all agents are deterministic rule-based (no GPU needed).
    """
    t0 = time.time()
    one_shot_fn = chat_fn if chat_fn is not None else model_fn

    # 1. Plan
    task_plan = _rule_based_planner(task) if demo else plan(task, one_shot_fn)

    # 2. Execute — use react_fn (chat template) when available
    if demo:
        result = execute(task, task_plan, model_fn=_rule_based_agent)
    else:
        result = execute(task, task_plan, react_fn=react_fn, model_fn=model_fn)

    # 3. Critique
    crit = (
        _rule_based_critic(task, result.final_answer)
        if demo
        else critique(task, result.final_answer, one_shot_fn)
    )

    retried = False

    # 4. Retry once with critic feedback injected into the executor context
    if not crit.approved:
        feedback_suffix = f"\n\nCRITIC FEEDBACK: {crit.feedback}\nPlease correct your answer."

        if demo:
            def feedback_exec(prompt: str) -> str:
                return _rule_based_agent(prompt + feedback_suffix)
            retry_result = execute(task, task_plan, model_fn=feedback_exec)
        elif react_fn is not None:
            saved_react_fn = react_fn
            def feedback_react(messages: list) -> str:
                # Append critic note to the last user message before generating
                augmented = messages[:-1] + [
                    {"role": "user",
                     "content": messages[-1]["content"] + feedback_suffix}
                ] if messages else messages
                return saved_react_fn(augmented)
            retry_result = execute(task, task_plan, react_fn=feedback_react)
        else:
            def feedback_model_fn(prompt: str) -> str:
                return model_fn(prompt + feedback_suffix)
            retry_result = execute(task, task_plan, model_fn=feedback_model_fn)

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


def eval_multi_agent(
    tasks: list[Task],
    model_fn: Callable[[str], str] | None = None,
    chat_fn: Callable[[str], str] | None = None,
    react_fn: Callable[[list], str] | None = None,
    demo: bool = False,
) -> dict:
    if demo and model_fn is None:
        model_fn = _rule_based_agent
    results = [
        run_multi_agent(t, model_fn=model_fn, chat_fn=chat_fn, react_fn=react_fn, demo=demo)
        for t in tasks
    ]
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

def compare(
    demo: bool = True,
    model_fn: Callable[[str], str] | None = None,
    chat_fn: Callable[[str], str] | None = None,
    react_fn: Callable[[list], str] | None = None,
) -> None:
    print("Single-agent (ReAct only):")
    if demo:
        single = eval_tool_use(model_fn=_rule_based_agent)
    else:
        single = eval_tool_use(react_fn=react_fn)
    print(f"  task_success={single['task_success']}  "
          f"tool_accuracy={single['tool_accuracy']}  "
          f"avg_steps={single['avg_steps_per_task']}")

    print("\nMulti-agent (Planner → Executor → Critic):")
    multi = eval_multi_agent(
        TASKS, model_fn=model_fn, chat_fn=chat_fn, react_fn=react_fn, demo=demo
    )
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
    ap.add_argument("--mode", choices=["demo", "hf"], default="demo",
                    help="demo=rule-based (no GPU); hf=HuggingFace model")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--compare", action="store_true",
                    help="Print single-agent vs multi-agent delta")
    args = ap.parse_args()

    demo = args.mode == "demo"

    if demo:
        model_fn = _rule_based_agent
        chat_fn = react_fn = None
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, device_map="auto"
        )
        model_fn = None
        react_fn = build_react_fn(mdl, tok)   # chat-template multi-turn — for Executor
        chat_fn  = build_chat_fn(mdl, tok)    # chat-template one-shot — for Planner + Critic
        print(f"Loaded {args.model} on {next(mdl.parameters()).device}")

    if args.compare:
        compare(demo=demo, model_fn=model_fn, chat_fn=chat_fn, react_fn=react_fn)
    else:
        results = eval_multi_agent(
            TASKS, model_fn=model_fn, chat_fn=chat_fn, react_fn=react_fn, demo=demo
        )
        print(json.dumps({k: v for k, v in results.items() if k != "results"}, indent=2))
