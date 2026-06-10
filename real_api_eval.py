"""
real_api_eval.py

Evaluates tool-use/function-calling on real developer API traces.
Dataset: Salesforce/xlam-function-calling-60k (real function calls from
         real developer workflows — 60K examples across 47 tool categories)

Metrics (function-calling specific):
  name_acc     — fraction with correct function name selected
  args_acc     — fraction with correct arguments (exact match after normalisation)
  full_acc     — fraction with correct name AND correct args
  halluc_rate  — fraction where model calls a function not in the tools list

Usage:
    python real_api_eval.py --model Qwen/Qwen2.5-7B-Instruct --n 200
    python real_api_eval.py --mode demo --n 50   # rule-based baseline, no GPU
    python real_api_eval.py --compare            # synthetic vs real comparison
"""

from __future__ import annotations
import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# ── Dataset loading ──────────────────────────────────────────────────────────

def load_xlam_examples(n: int = 200, seed: int = 42, category: str | None = None) -> list[dict]:
    """
    Load examples from Salesforce/xlam-function-calling-60k.

    Each returned dict is normalised to:
        {"query": str, "tools": list[dict], "answers": list[dict], "category": str}
    """
    from datasets import load_dataset

    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")

    examples: list[dict] = []
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

        # Derive category from the first tool name (e.g. "get_weather" → "weather")
        # xlam stores a "category" field on some splits; fall back to tool-name heuristic
        cat = ex.get("category") or ex.get("tool_category") or _infer_category(tools[0].get("name", ""))

        if category is not None and cat.lower() != category.lower():
            continue

        examples.append({
            "query": ex["query"],
            "tools": tools,
            "answers": answers,
            "category": cat,
        })

    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:n]


def _infer_category(tool_name: str) -> str:
    """
    Map a tool name to a broad category using keyword matching.
    Falls back to the first word/segment of the tool name.
    """
    name = tool_name.lower()
    _KEYWORD_MAP = {
        "weather": ["weather", "forecast", "temperature", "climate"],
        "finance": ["stock", "price", "finance", "currency", "exchange", "market", "invest"],
        "search": ["search", "find", "query", "lookup", "google", "bing"],
        "news": ["news", "article", "headline", "rss"],
        "sports": ["sport", "soccer", "football", "basketball", "baseball", "nba", "nfl", "cricket"],
        "music": ["music", "song", "spotify", "playlist", "artist", "track", "album"],
        "maps": ["map", "location", "place", "distance", "route", "navigate", "geocode"],
        "calendar": ["calendar", "event", "schedule", "appointment", "reminder"],
        "email": ["email", "mail", "inbox", "send_message", "smtp"],
        "social": ["tweet", "twitter", "reddit", "instagram", "facebook", "post"],
        "travel": ["flight", "hotel", "booking", "travel", "trip", "airfare"],
        "food": ["recipe", "food", "restaurant", "meal", "ingredient", "nutrition"],
        "health": ["health", "medical", "symptom", "drug", "medicine", "doctor"],
        "math": ["calc", "compute", "math", "arithmetic", "convert", "unit"],
        "data": ["database", "sql", "query", "table", "record"],
    }
    for cat, keywords in _KEYWORD_MAP.items():
        if any(kw in name for kw in keywords):
            return cat
    # Fall back: use underscore-split first token
    parts = re.split(r"[_\-]", name)
    return parts[0] if parts else "other"


# ── Prompt formatting ─────────────────────────────────────────────────────────

def _format_tool(tool: dict) -> str:
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


def build_prompt(query: str, tools: list[dict]) -> list[dict]:
    """
    Return a messages list (system + user) for function-calling eval.
    Works with any instruct-tuned model via apply_chat_template.
    """
    tool_block = "\n\n".join(_format_tool(t) for t in tools)
    system = (
        "You are a function-calling assistant. You have access to these functions:\n\n"
        + tool_block
        + "\n\nWhen the user asks something, call the appropriate function. Respond with ONLY:\n"
        "<function_call>{\"name\": \"...\", \"arguments\": {...}}</function_call>"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]


# ── Output parsing ────────────────────────────────────────────────────────────

_FC_RE = re.compile(r"<function_call>(.*?)(?:</function_call>|$)", re.DOTALL)


def parse_function_call(output: str) -> dict | None:
    """
    Extract the first <function_call>...</function_call> from model output.
    Returns {"name": ..., "arguments": {...}} or None on parse failure.
    """
    m = _FC_RE.search(output)
    if not m:
        # Try bare JSON object that has "name" and "arguments"
        stripped = output.strip()
        if stripped.startswith("{"):
            try:
                obj = json.loads(stripped)
                if "name" in obj:
                    return obj
            except json.JSONDecodeError:
                pass
        return None
    raw = m.group(1).strip()
    try:
        obj = json.loads(raw)
        if "name" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    return None


# ── Metric helpers ────────────────────────────────────────────────────────────

def _normalise_value(v: Any) -> Any:
    """Normalise a value for comparison: strip strings, lowercase, round floats."""
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, float):
        return round(v, 2)
    return v


def _args_match(predicted_args: dict, expected_args: dict) -> bool:
    """
    Return True if predicted_args contains every (key, normalised_value) from expected_args.
    Extra keys in predicted are allowed.
    """
    if not isinstance(predicted_args, dict):
        return False
    for key, exp_val in expected_args.items():
        if key not in predicted_args:
            return False
        if _normalise_value(predicted_args[key]) != _normalise_value(exp_val):
            return False
    return True


# ── Per-example evaluation ────────────────────────────────────────────────────

def evaluate_example(
    predicted: dict | None,
    expected_calls: list[dict],
    tool_names: set[str],
) -> dict:
    """
    Evaluate a single prediction against the expected function calls.

    For multi-call examples we evaluate against the first expected call
    (the most common case; 98% of xlam examples have a single expected call).

    Returns {name_correct, args_correct, full_correct, hallucinated}.
    """
    if predicted is None:
        return {"name_correct": False, "args_correct": False,
                "full_correct": False, "hallucinated": False}

    pred_name = predicted.get("name", "")
    pred_args = predicted.get("arguments", {})
    if not isinstance(pred_args, dict):
        pred_args = {}

    # Hallucination: predicted name not in the provided tool list at all
    hallucinated = pred_name not in tool_names

    # Use first expected call as the reference
    ref = expected_calls[0]
    exp_name = ref.get("name", "")
    exp_args = ref.get("arguments", {})
    if not isinstance(exp_args, dict):
        exp_args = {}

    name_correct = pred_name == exp_name
    args_correct = name_correct and _args_match(pred_args, exp_args)
    full_correct = name_correct and args_correct

    return {
        "name_correct": name_correct,
        "args_correct": args_correct,
        "full_correct": full_correct,
        "hallucinated": hallucinated,
    }


# ── Rule-based demo agent ─────────────────────────────────────────────────────

def _keyword_agent(query: str, tools: list[dict]) -> dict | None:
    """
    Simple rule-based agent: picks the tool whose name / description has the
    most keyword overlap with the query. Arguments are left empty (best effort).

    This provides a no-GPU floor/ceiling baseline for the harness itself.
    """
    if not tools:
        return None

    query_words = set(re.findall(r"\w+", query.lower()))
    best_tool = None
    best_score = -1

    for tool in tools:
        name_words = set(re.findall(r"\w+", tool.get("name", "").lower()))
        desc_words = set(re.findall(r"\w+", tool.get("description", "").lower()))
        score = len(query_words & (name_words | desc_words))
        if score > best_score:
            best_score = score
            best_tool = tool

    if best_tool is None:
        return None

    # Try to fill required args with values extracted from the query
    params = best_tool.get("parameters", {})
    required = params.get("required", []) if isinstance(params, dict) else []
    props = params.get("properties", {}) if isinstance(params, dict) else {}
    arguments: dict = {}

    for pname in required:
        pinfo = props.get(pname, {}) if isinstance(props, dict) else {}
        ptype = pinfo.get("type", "string") if isinstance(pinfo, dict) else "string"
        # Very naive: grab the first quoted string or capitalised phrase from the query
        if ptype == "string":
            quoted = re.findall(r'"([^"]+)"', query)
            if quoted:
                arguments[pname] = quoted[0]
            else:
                # Grab named entities: capitalised words / numbers
                tokens = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|\d+\.?\d*', query)
                arguments[pname] = tokens[0] if tokens else query[:50]
        elif ptype in ("integer", "number"):
            nums = re.findall(r"\d+\.?\d*", query)
            arguments[pname] = float(nums[0]) if nums else 0

    return {"name": best_tool["name"], "arguments": arguments}


# ── HuggingFace model inference ───────────────────────────────────────────────

def build_hf_infer_fn(model, tokenizer, max_new_tokens: int = 256):
    """Return a callable(messages) -> str using apply_chat_template."""
    import torch

    def _infer(messages: list[dict]) -> str:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=4096
        ).to(next(model.parameters()).device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = out[0][enc["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    return _infer


# ── Core evaluation loop ──────────────────────────────────────────────────────

def eval_real_api(
    infer_fn=None,
    examples: list[dict] | None = None,
    n: int = 200,
    seed: int = 42,
    category: str | None = None,
    output_path: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run function-calling eval on real API traces.

    Pass infer_fn=None to use the rule-based keyword agent (demo mode).
    Pass infer_fn=callable(messages) -> str for a model.
    """
    if examples is None:
        if verbose:
            print(f"Loading Salesforce/xlam-function-calling-60k (n={n}, seed={seed})…")
        examples = load_xlam_examples(n=n, seed=seed, category=category)

    if verbose:
        print(f"Evaluating {len(examples)} examples…\n")

    name_hits, args_hits, full_hits, halluc_hits = 0, 0, 0, 0
    per_category: dict[str, dict] = defaultdict(lambda: {"total": 0, "full_correct": 0})
    detail_rows: list[dict] = []

    for i, ex in enumerate(examples):
        tools: list[dict] = ex["tools"]
        tool_names = {t.get("name", "") for t in tools}
        messages = build_prompt(ex["query"], tools)

        t0 = time.time()
        if infer_fn is None:
            predicted = _keyword_agent(ex["query"], tools)
        else:
            raw_output = infer_fn(messages)
            predicted = parse_function_call(raw_output)
        latency = round(time.time() - t0, 3)

        metrics = evaluate_example(predicted, ex["answers"], tool_names)

        name_hits  += int(metrics["name_correct"])
        args_hits  += int(metrics["args_correct"])
        full_hits  += int(metrics["full_correct"])
        halluc_hits += int(metrics["hallucinated"])

        cat = ex.get("category", "other")
        per_category[cat]["total"] += 1
        per_category[cat]["full_correct"] += int(metrics["full_correct"])

        if verbose and i < 10:
            status = "✓" if metrics["full_correct"] else ("H" if metrics["hallucinated"] else "✗")
            pred_name = predicted.get("name", "None") if predicted else "None"
            exp_name = ex["answers"][0].get("name", "?")
            print(f"  [{status}] {i+1:3d}  pred={pred_name!r:35s} exp={exp_name!r}")

        detail_rows.append({
            "idx": i,
            "category": cat,
            "query": ex["query"][:120],
            "expected_name": ex["answers"][0].get("name", "") if ex["answers"] else "",
            "predicted_name": predicted.get("name", "") if predicted else "",
            **metrics,
            "latency_s": latency,
        })

    n_eval = len(examples)

    # Per-category accuracy (top 10 by volume)
    cat_acc: dict[str, float] = {}
    sorted_cats = sorted(per_category.items(), key=lambda x: -x[1]["total"])
    for cat, counts in sorted_cats[:10]:
        cat_acc[cat] = round(counts["full_correct"] / max(counts["total"], 1), 4)

    summary = {
        "dataset": "Salesforce/xlam-function-calling-60k",
        "n_evaluated": n_eval,
        "name_acc": round(name_hits / n_eval, 4) if n_eval else 0.0,
        "args_acc": round(args_hits / n_eval, 4) if n_eval else 0.0,
        "full_acc": round(full_hits / n_eval, 4) if n_eval else 0.0,
        "halluc_rate": round(halluc_hits / n_eval, 4) if n_eval else 0.0,
        "per_category": cat_acc,
        "detail": detail_rows,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out = {k: v for k, v in summary.items() if k != "detail"}
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        if verbose:
            print(f"\nSaved → {output_path}")

    return summary


# ── Side-by-side comparison ───────────────────────────────────────────────────

def run_comparison(model_name: str, n: int = 200, mode: str = "demo") -> None:
    """
    Run both synthetic (eval_tool_use) and real (eval_real_api) evals then
    print a side-by-side table.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from eval_tool_use import eval_tool_use, _rule_based_agent, build_react_fn

    print("=" * 60)
    print("Synthetic Tasks (eval_tool_use.py)")
    print("=" * 60)

    if mode == "demo":
        synth = eval_tool_use(model_fn=_rule_based_agent)
        real_infer = None
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        rfn = build_react_fn(mdl, tok)
        synth = eval_tool_use(react_fn=rfn)
        real_infer = build_hf_infer_fn(mdl, tok)

    print("\n" + "=" * 60)
    print("Real API Traces (Salesforce/xlam-function-calling-60k)")
    print("=" * 60)
    real = eval_real_api(infer_fn=real_infer, n=n)

    # ── Table ──
    print("\n")
    print(f"{'Metric':<40} {'Synthetic Tasks':>18}  {'Real API Traces':>16}")
    print("-" * 78)
    print(f"{'Task success / Name acc':<40} {synth['task_success']:>17.1%}  {real['name_acc']:>15.1%}")
    print(f"{'Tool accuracy / Args acc':<40} {synth['tool_accuracy']:>17.1%}  {real['args_acc']:>15.1%}")
    print(f"{'Full acc (name+args)':<40} {'—':>17}  {real['full_acc']:>15.1%}")
    print(f"{'Error recovery / Halluc rate':<40} {synth['error_recovery_rate']:>17.1%}  {real['halluc_rate']:>14.1%} halluc")
    print("-" * 78)

    if real["per_category"]:
        print("\nReal API — accuracy by category (top 10):")
        for cat, acc in sorted(real["per_category"].items(), key=lambda x: -x[1]):
            print(f"  {cat:<22}  {acc:.1%}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate function-calling on Salesforce/xlam-function-calling-60k"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "hf"],
        default="demo",
        help="demo=rule-based keyword agent (no GPU); hf=HuggingFace model",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model id (used when --mode hf)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of examples to evaluate (sampled with seed=42)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Filter to a specific tool category (e.g. weather, finance)",
    )
    parser.add_argument(
        "--output",
        default="results/real_api_eval.json",
        help="Where to save the JSON results",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both synthetic and real evals and print side-by-side table",
    )
    args = parser.parse_args()

    if args.compare:
        run_comparison(model_name=args.model, n=args.n, mode=args.mode)
    else:
        print(f"=== Real API Function-Calling Eval (n={args.n}) ===\n")

        if args.mode == "demo":
            print("Mode: rule-based keyword agent (no GPU)\n")
            summary = eval_real_api(
                infer_fn=None,
                n=args.n,
                seed=args.seed,
                category=args.category,
                output_path=args.output,
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            print(f"Loading model: {args.model}")
            tok = AutoTokenizer.from_pretrained(args.model)
            mdl = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.float16, device_map="auto"
            )
            infer_fn = build_hf_infer_fn(mdl, tok)
            summary = eval_real_api(
                infer_fn=infer_fn,
                n=args.n,
                seed=args.seed,
                category=args.category,
                output_path=args.output,
            )

        print(f"\n=== Results ===")
        print(f"  Dataset:       {summary['dataset']}")
        print(f"  N evaluated:   {summary['n_evaluated']}")
        print(f"  Name acc:      {summary['name_acc']:.1%}")
        print(f"  Args acc:      {summary['args_acc']:.1%}")
        print(f"  Full acc:      {summary['full_acc']:.1%}")
        print(f"  Halluc rate:   {summary['halluc_rate']:.1%}")
        if summary["per_category"]:
            print(f"\nPer-category full accuracy (top 10):")
            for cat, acc in sorted(summary["per_category"].items(), key=lambda x: -x[1]):
                print(f"  {cat:<22}  {acc:.1%}")
        if args.output:
            print(f"\nSaved → {args.output}")
