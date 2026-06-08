"""
Runs lm-eval-harness on a model and compares results with the custom eval results.
Requires: pip install lm-eval
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_lm_eval(
    model_name: str,
    tasks: list[str],
    num_fewshot: int = 0,
    limit: int = 500,
    output_path: str = "results/lm_eval_results.json",
) -> dict:
    """
    Runs lm-eval-harness via CLI and parses the JSON results.
    tasks: e.g. ["gsm8k", "humaneval", "mmlu", "hellaswag"]
    Returns parsed results dict, or empty dict on failure.
    """
    try:
        import lm_eval  # noqa: F401
    except ImportError:
        print(
            "lm-eval is not installed.\n"
            "Install it with:\n"
            "  pip install lm-eval\n"
            "Or from source:\n"
            "  pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git"
        )
        return {}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    tasks_str = ",".join(tasks)
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_name}",
        "--tasks", tasks_str,
        "--num_fewshot", str(num_fewshot),
        "--limit", str(limit),
        "--output_path", output_path,
        "--log_samples",
    ]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"lm_eval exited with code {result.returncode}")
        return {}

    # lm-eval writes a nested directory; find the JSON
    out_p = Path(output_path)
    json_path = None
    if out_p.is_file():
        json_path = out_p
    elif out_p.is_dir():
        hits = sorted(out_p.rglob("*.json"))
        if hits:
            json_path = hits[-1]

    if json_path is None or not json_path.exists():
        print(f"Could not locate output JSON under {output_path}")
        return {}

    with open(json_path) as f:
        data = json.load(f)

    print(f"Loaded lm-eval results from {json_path}")
    return data


def _extract_metric(results: dict, task: str) -> float | None:
    """
    Pull the primary accuracy metric for a given task name from the lm-eval
    results dict.  lm-eval key format: results[task]["metric,none"] or
    results[task]["metric"].
    """
    task_data = results.get("results", results).get(task)
    if task_data is None:
        return None

    # Priority order of metric keys used by common tasks
    for key in (
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "exact_match,none",
        "acc_norm,none",
        "acc,none",
        "pass@1,none",
        "exact_match",
        "acc_norm",
        "acc",
        "pass@1",
    ):
        if key in task_data:
            val = task_data[key]
            # lm-eval sometimes returns values in [0,1]; some older versions in [0,100]
            if isinstance(val, (int, float)):
                return float(val) if val <= 1.0 else float(val) / 100.0
    return None


def compare_with_custom(
    lm_eval_results: dict,
    custom_csv: str = "metrics_summary.csv",
) -> pd.DataFrame:
    """
    Build a side-by-side comparison table.

    lm-eval results are keyed by task name (gsm8k, humaneval, …).
    Custom CSV has columns: Model, pass@1, Latency.
    We map custom pass@1 of the aggregate row to HumanEval.
    """
    custom_p = Path(custom_csv)
    if not custom_p.exists():
        print(f"Custom CSV not found: {custom_csv}")
        custom_df = pd.DataFrame()
    else:
        custom_df = pd.read_csv(custom_csv)

    # Normalise custom pass@1 column to float
    if not custom_df.empty and "pass@1" in custom_df.columns:
        custom_df["pass@1"] = (
            custom_df["pass@1"]
            .astype(str)
            .str.rstrip("%")
            .astype(float)
            .div(100)
        )

    rows = []
    task_map = {
        "gsm8k":      "GSM8K",
        "humaneval":  "HumanEval",
        "mmlu":       "MMLU",
        "hellaswag":  "HellaSwag",
        "arc_easy":   "ARC-Easy",
        "arc_challenge": "ARC-Challenge",
    }

    all_tasks = set()
    if lm_eval_results:
        all_tasks.update(lm_eval_results.get("results", lm_eval_results).keys())
    all_tasks.update(task_map.keys())

    for task_key in sorted(all_tasks):
        lm_val = _extract_metric(lm_eval_results, task_key) if lm_eval_results else None

        # Pull from custom CSV — use best-scoring model row for comparison
        custom_val = None
        if not custom_df.empty and task_key == "humaneval" and "pass@1" in custom_df.columns:
            custom_val = custom_df["pass@1"].max()

        label = task_map.get(task_key, task_key)
        rows.append({
            "Task": label,
            "lm-eval score": f"{lm_val:.3f}" if lm_val is not None else "—",
            "Custom eval score": f"{custom_val:.3f}" if custom_val is not None else "—",
            "Delta": (
                f"{lm_val - custom_val:+.3f}"
                if lm_val is not None and custom_val is not None
                else "—"
            ),
        })

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run lm-eval-harness and optionally compare with custom eval results."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--tasks", nargs="+", default=["gsm8k", "humaneval"],
        help="lm-eval task names, e.g. gsm8k humaneval mmlu hellaswag",
    )
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--output", default="results/lm_eval_results.json",
        help="Path for lm-eval JSON output",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="After running, print comparison table against metrics_summary.csv",
    )
    parser.add_argument(
        "--custom_csv", default="metrics_summary.csv",
        help="Path to custom eval CSV (used with --compare)",
    )
    args = parser.parse_args()

    lm_results = run_lm_eval(
        model_name=args.model,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        output_path=args.output,
    )

    if not lm_results:
        print("No lm-eval results to display.")
        return

    # Print raw per-task summary
    print("\n=== lm-eval Results ===")
    for task in args.tasks:
        val = _extract_metric(lm_results, task)
        score_str = f"{val:.3f}" if val is not None else "n/a"
        print(f"  {task:30s} {score_str}")

    if args.compare:
        print("\n=== Comparison: lm-eval vs Custom Eval ===")
        cmp_df = compare_with_custom(lm_results, custom_csv=args.custom_csv)
        print(cmp_df.to_string(index=False))

        cmp_path = Path(args.output).parent / "comparison_table.csv"
        cmp_df.to_csv(cmp_path, index=False)
        print(f"\nSaved comparison table → {cmp_path}")


if __name__ == "__main__":
    main()
