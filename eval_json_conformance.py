"""
eval_json_conformance.py

JSON schema conformance evaluator.

Prompts a model to produce structured JSON output matching a given schema.
Measures four error modes that matter in production API usage:

  strict_acc      — valid JSON + all required fields + correct types
  partial_acc     — valid JSON + required fields present but ≥1 type mismatch
  halluc_rate     — fraction of outputs with extra fields not in the schema (hallucinated keys)
  missing_rate    — fraction of outputs missing ≥1 required field

Schema suite: 10 real-world schemas across API-relevant categories:
  user_profile, api_response, tool_call, search_result, code_review,
  calendar_event, product_listing, error_response, model_completion, agent_action

Usage:
    # Demo mode (rule-based agent, no GPU)
    python eval_json_conformance.py --mode demo

    # Real model
    CUDA_VISIBLE_DEVICES=0 python eval_json_conformance.py \\
        --model Qwen/Qwen2.5-7B-Instruct --n 50

    # Save results
    python eval_json_conformance.py --mode demo --output results/json_conformance.json
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False
    print("WARNING: jsonschema not installed. Run: pip install jsonschema")
    print("         strict_acc will always be False without it.\n")

# ── Schema definitions ────────────────────────────────────────────────────────

SCHEMAS: dict[str, dict] = {
    "user_profile": {
        "type": "object",
        "required": ["user_id", "name", "email", "created_at"],
        "properties": {
            "user_id":    {"type": "integer"},
            "name":       {"type": "string"},
            "email":      {"type": "string"},
            "age":        {"type": "integer", "minimum": 0},
            "created_at": {"type": "string", "format": "date-time"},
            "tags":       {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    },
    "api_response": {
        "type": "object",
        "required": ["status", "data", "request_id"],
        "properties": {
            "status":     {"type": "string", "enum": ["success", "error", "pending"]},
            "data":       {"type": "object"},
            "request_id": {"type": "string"},
            "error":      {"type": "string"},
            "timestamp":  {"type": "number"},
        },
        "additionalProperties": False,
    },
    "tool_call": {
        "type": "object",
        "required": ["name", "arguments"],
        "properties": {
            "name":      {"type": "string"},
            "arguments": {"type": "object"},
            "call_id":   {"type": "string"},
        },
        "additionalProperties": False,
    },
    "search_result": {
        "type": "object",
        "required": ["title", "url", "snippet", "score"],
        "properties": {
            "title":    {"type": "string"},
            "url":      {"type": "string"},
            "snippet":  {"type": "string"},
            "score":    {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "metadata": {"type": "object"},
        },
        "additionalProperties": False,
    },
    "code_review": {
        "type": "object",
        "required": ["verdict", "severity", "comments"],
        "properties": {
            "verdict":   {"type": "string", "enum": ["approve", "request_changes", "comment"]},
            "severity":  {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            "comments":  {"type": "array", "items": {"type": "string"}},
            "line_refs": {"type": "array", "items": {"type": "integer"}},
        },
        "additionalProperties": False,
    },
    "calendar_event": {
        "type": "object",
        "required": ["title", "start_time", "end_time", "attendees"],
        "properties": {
            "title":       {"type": "string"},
            "start_time":  {"type": "string"},
            "end_time":    {"type": "string"},
            "attendees":   {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "location":    {"type": "string"},
            "description": {"type": "string"},
            "recurrence":  {"type": "string"},
        },
        "additionalProperties": False,
    },
    "product_listing": {
        "type": "object",
        "required": ["product_id", "name", "price", "in_stock"],
        "properties": {
            "product_id":   {"type": "string"},
            "name":         {"type": "string"},
            "price":        {"type": "number", "minimum": 0},
            "in_stock":     {"type": "boolean"},
            "category":     {"type": "string"},
            "rating":       {"type": "number", "minimum": 0.0, "maximum": 5.0},
            "review_count": {"type": "integer", "minimum": 0},
        },
        "additionalProperties": False,
    },
    "error_response": {
        "type": "object",
        "required": ["code", "message", "type"],
        "properties": {
            "code":        {"type": "integer"},
            "message":     {"type": "string"},
            "type":        {"type": "string", "enum": ["validation_error", "auth_error", "rate_limit", "server_error", "not_found"]},
            "details":     {"type": "object"},
            "retry_after": {"type": "integer"},
        },
        "additionalProperties": False,
    },
    "model_completion": {
        "type": "object",
        "required": ["id", "model", "choices", "usage"],
        "properties": {
            "id":      {"type": "string"},
            "model":   {"type": "string"},
            "choices": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["index", "message", "finish_reason"],
                    "properties": {
                        "index":         {"type": "integer"},
                        "message":       {"type": "object"},
                        "finish_reason": {"type": "string"},
                    },
                },
            },
            "usage": {
                "type": "object",
                "required": ["prompt_tokens", "completion_tokens"],
                "properties": {
                    "prompt_tokens":     {"type": "integer"},
                    "completion_tokens": {"type": "integer"},
                    "total_tokens":      {"type": "integer"},
                },
            },
        },
    },
    "agent_action": {
        "type": "object",
        "required": ["action_type", "payload", "reasoning"],
        "properties": {
            "action_type": {"type": "string", "enum": ["tool_call", "respond", "clarify", "delegate", "terminate"]},
            "payload":     {"type": "object"},
            "reasoning":   {"type": "string"},
            "confidence":  {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "additionalProperties": False,
    },
}

# ── Hand-authored reference outputs (demo mode) ───────────────────────────────

DEMO_OUTPUTS: dict[str, dict] = {
    "user_profile": {
        "user_id": 42,
        "name": "Alice Nguyen",
        "email": "alice@example.com",
        "age": 31,
        "created_at": "2024-03-15T10:30:00Z",
        "tags": ["beta_user", "premium"],
    },
    "api_response": {
        "status": "success",
        "data": {"result": "ok"},
        "request_id": "req_abc123",
        "timestamp": 1710000000.0,
    },
    "tool_call": {
        "name": "search_web",
        "arguments": {"query": "latest AI benchmarks", "num_results": 5},
        "call_id": "call_x9f2",
    },
    "search_result": {
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/abs/1706.03762",
        "snippet": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
        "score": 0.97,
        "metadata": {"year": 2017, "citations": 80000},
    },
    "code_review": {
        "verdict": "request_changes",
        "severity": "medium",
        "comments": ["Missing error handling in network call", "Variable name `d` is not descriptive"],
        "line_refs": [42, 87],
    },
    "calendar_event": {
        "title": "Team Sync — Q2 Planning",
        "start_time": "2024-04-01T14:00:00Z",
        "end_time": "2024-04-01T15:00:00Z",
        "attendees": ["alice@example.com", "bob@example.com"],
        "location": "Conference Room B",
        "description": "Quarterly planning session",
        "recurrence": "RRULE:FREQ=WEEKLY;BYDAY=MO",
    },
    "product_listing": {
        "product_id": "SKU-7890",
        "name": "Mechanical Keyboard MK500",
        "price": 129.99,
        "in_stock": True,
        "category": "Electronics",
        "rating": 4.6,
        "review_count": 312,
    },
    "error_response": {
        "code": 429,
        "message": "Rate limit exceeded. Please wait before retrying.",
        "type": "rate_limit",
        "details": {"limit": 100, "window_seconds": 60},
        "retry_after": 30,
    },
    "model_completion": {
        "id": "cmpl-9xKf3Lm",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "The answer is 42."},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 18,
            "completion_tokens": 7,
            "total_tokens": 25,
        },
    },
    "agent_action": {
        "action_type": "tool_call",
        "payload": {"tool": "web_search", "query": "current weather in Tokyo"},
        "reasoning": "The user asked about weather, so I need to fetch live data.",
        "confidence": 0.91,
    },
}

# ── Prompt construction ───────────────────────────────────────────────────────

def build_prompt(schema_name: str, schema: dict) -> str:
    schema_str = json.dumps(schema, indent=2)
    return (
        f"Generate a JSON object matching the following JSON Schema (schema name: {schema_name}).\n"
        f"Return ONLY the JSON with no additional text, explanation, or markdown fences.\n\n"
        f"Schema:\n{schema_str}"
    )

# ── Evaluation logic ──────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def evaluate_output(output: str, schema: dict) -> dict:
    """
    Evaluate a single model output against a schema.

    Returns:
        valid_json        — whether the output parses as JSON
        strict            — passes jsonschema.validate() (requires valid_json)
        partial           — valid JSON + no missing required fields (may have type issues)
        hallucinated_keys — keys in output not declared in schema properties
        missing_keys      — required keys absent from the output
    """
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    # 1. Parse JSON
    cleaned = _strip_fences(output)
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return {
            "valid_json": False,
            "strict": False,
            "partial": False,
            "hallucinated_keys": [],
            "missing_keys": list(required),
        }

    if not isinstance(parsed, dict):
        return {
            "valid_json": True,
            "strict": False,
            "partial": False,
            "hallucinated_keys": [],
            "missing_keys": list(required),
        }

    # 2. Missing required fields
    missing_keys = [k for k in required if k not in parsed]

    # 3. Hallucinated keys (keys not declared in properties)
    hallucinated_keys = [k for k in parsed if k not in properties] if properties else []

    # 4. Strict: passes full jsonschema validation
    strict = False
    if _HAS_JSONSCHEMA:
        try:
            jsonschema.validate(instance=parsed, schema=schema)
            strict = True
        except jsonschema.ValidationError:
            strict = False
    else:
        # Fallback: strict only if no missing keys and no hallucinated keys
        strict = (len(missing_keys) == 0 and len(hallucinated_keys) == 0)

    # 5. Partial: valid JSON + no missing required fields (type mismatches OK)
    partial = len(missing_keys) == 0

    return {
        "valid_json": True,
        "strict": strict,
        "partial": partial,
        "hallucinated_keys": hallucinated_keys,
        "missing_keys": missing_keys,
    }

# ── Per-schema result accumulator ─────────────────────────────────────────────

@dataclass
class SchemaResult:
    schema_name: str
    n: int = 0
    strict_count: int = 0
    partial_count: int = 0
    halluc_count: int = 0      # outputs with ≥1 hallucinated key
    missing_count: int = 0     # outputs with ≥1 missing required key
    all_hallucinated: list[str] = field(default_factory=list)
    all_missing: list[str] = field(default_factory=list)

    def update(self, result: dict) -> None:
        self.n += 1
        if result["strict"]:
            self.strict_count += 1
        if result["partial"]:
            self.partial_count += 1
        if result["hallucinated_keys"]:
            self.halluc_count += 1
            self.all_hallucinated.extend(result["hallucinated_keys"])
        if result["missing_keys"]:
            self.missing_count += 1
            self.all_missing.extend(result["missing_keys"])

    @property
    def strict_acc(self) -> float:
        return self.strict_count / self.n if self.n else 0.0

    @property
    def partial_acc(self) -> float:
        return self.partial_count / self.n if self.n else 0.0

    @property
    def halluc_rate(self) -> float:
        return self.halluc_count / self.n if self.n else 0.0

    @property
    def missing_rate(self) -> float:
        return self.missing_count / self.n if self.n else 0.0

# ── Demo agent ────────────────────────────────────────────────────────────────

def _demo_agent(schema_name: str) -> str:
    """Return the hand-authored perfect JSON for each schema."""
    return json.dumps(DEMO_OUTPUTS[schema_name])

# ── HuggingFace model helpers ─────────────────────────────────────────────────

def build_hf_fn(model, tokenizer, max_new_tokens: int = 512):
    """Wrap a HuggingFace instruct model into a callable(prompt) -> str."""
    import torch

    def _call(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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

    return _call

# ── Main evaluation loop ──────────────────────────────────────────────────────

def eval_json_conformance(
    model_fn=None,
    n: int = 1,
    output_path: str | None = None,
) -> dict:
    """
    Run JSON conformance evaluation across all schemas.

    Args:
        model_fn:    callable(schema_name: str, prompt: str) -> str
                     For demo mode pass a wrapper around _demo_agent.
        n:           number of samples per schema.
        output_path: if set, write JSON results to this path.

    Returns:
        Summary dict with per-schema and overall metrics.
    """
    schema_results: dict[str, SchemaResult] = {
        name: SchemaResult(schema_name=name) for name in SCHEMAS
    }

    t_start = time.time()

    for schema_name, schema in SCHEMAS.items():
        prompt = build_prompt(schema_name, schema)
        for _ in range(n):
            raw = model_fn(schema_name, prompt)
            result = evaluate_output(raw, schema)
            schema_results[schema_name].update(result)

    elapsed = round(time.time() - t_start, 2)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    all_hallucinated: list[str] = []
    all_missing: list[str] = []
    total_strict = total_partial = total_halluc = total_missing = total_n = 0

    per_schema: list[dict] = []
    for name, sr in schema_results.items():
        all_hallucinated.extend(sr.all_hallucinated)
        all_missing.extend(sr.all_missing)
        total_strict  += sr.strict_count
        total_partial += sr.partial_count
        total_halluc  += sr.halluc_count
        total_missing += sr.missing_count
        total_n       += sr.n
        per_schema.append({
            "schema":       name,
            "n":            sr.n,
            "strict_acc":   round(sr.strict_acc, 4),
            "partial_acc":  round(sr.partial_acc, 4),
            "halluc_rate":  round(sr.halluc_rate, 4),
            "missing_rate": round(sr.missing_rate, 4),
            "top_hallucinated": [k for k, _ in Counter(sr.all_hallucinated).most_common(5)],
            "top_missing":      [k for k, _ in Counter(sr.all_missing).most_common(5)],
        })

    overall = {
        "n":            total_n,
        "strict_acc":   round(total_strict  / total_n, 4) if total_n else 0.0,
        "partial_acc":  round(total_partial / total_n, 4) if total_n else 0.0,
        "halluc_rate":  round(total_halluc  / total_n, 4) if total_n else 0.0,
        "missing_rate": round(total_missing / total_n, 4) if total_n else 0.0,
    }

    top_hallucinated = [k for k, _ in Counter(all_hallucinated).most_common(10)]
    top_missing      = [k for k, _ in Counter(all_missing).most_common(10)]

    summary = {
        "elapsed_s":        elapsed,
        "overall":          overall,
        "per_schema":       per_schema,
        "top_hallucinated": top_hallucinated,
        "top_missing":      top_missing,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

    return summary


def print_results(summary: dict) -> None:
    overall   = summary["overall"]
    per_schema = summary["per_schema"]

    col_w = 20
    print("\n=== JSON Conformance Results ===\n")
    header = (
        f"{'Schema':{col_w}}  {'strict_acc':>10}  {'partial_acc':>11}  "
        f"{'halluc_rate':>11}  {'missing_rate':>12}  {'n':>5}"
    )
    print(header)
    print("-" * (len(header) + 4))

    for row in per_schema:
        print(
            f"{row['schema']:{col_w}}  "
            f"{row['strict_acc']:>9.1%}  "
            f"{row['partial_acc']:>10.1%}  "
            f"{row['halluc_rate']:>10.1%}  "
            f"{row['missing_rate']:>11.1%}  "
            f"{row['n']:>5}"
        )

    print("-" * (len(header) + 4))
    print(
        f"{'OVERALL':{col_w}}  "
        f"{overall['strict_acc']:>9.1%}  "
        f"{overall['partial_acc']:>10.1%}  "
        f"{overall['halluc_rate']:>10.1%}  "
        f"{overall['missing_rate']:>11.1%}  "
        f"{overall['n']:>5}"
    )

    if summary["top_hallucinated"]:
        print(f"\nMost common hallucinated keys: {summary['top_hallucinated']}")
    if summary["top_missing"]:
        print(f"Most common missing keys:      {summary['top_missing']}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JSON Schema Conformance Evaluator")
    parser.add_argument("--mode", choices=["demo", "hf"], default="demo",
                        help="demo=rule-based (no GPU); hf=HuggingFace model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model id (--mode hf only)")
    parser.add_argument("--n", type=int, default=1,
                        help="Samples per schema (demo defaults to 1; use ≥10 for real evals)")
    parser.add_argument("--output", default=None,
                        help="Path to write JSON results (e.g. results/json_conformance.json)")
    args = parser.parse_args()

    n_samples = args.n

    if args.mode == "demo":
        print(f"=== JSON Conformance Eval — demo mode ({len(SCHEMAS)} schemas × {n_samples} sample(s)) ===")

        def _demo_fn(schema_name: str, prompt: str) -> str:  # noqa: F811
            return _demo_agent(schema_name)

        summary = eval_json_conformance(model_fn=_demo_fn, n=n_samples, output_path=args.output)

    else:
        if n_samples == 1:
            n_samples = 50
            print(f"Note: --n not set, defaulting to {n_samples} samples per schema for hf mode.")

        print(f"=== JSON Conformance Eval — {args.model} ({len(SCHEMAS)} schemas × {n_samples} samples) ===")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tok = AutoTokenizer.from_pretrained(args.model)
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="auto"
        )
        hf_call = build_hf_fn(mdl, tok)

        def _hf_fn(schema_name: str, prompt: str) -> str:
            return hf_call(prompt)

        summary = eval_json_conformance(model_fn=_hf_fn, n=n_samples, output_path=args.output)

    print_results(summary)

    if args.output:
        print(f"\nSaved → {args.output}")
