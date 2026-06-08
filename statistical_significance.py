"""
Bootstrap confidence intervals for LLM benchmark scores.
Uses 10,000 bootstrap samples (fast on CPU).

Usage:
    python statistical_significance.py --results detailed_results.csv
    python statistical_significance.py --results detailed_results.csv --output results/statistical_report.json
    python statistical_significance.py --power --effect_size 0.02
"""
import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(
    scores: list[float],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
) -> dict:
    """
    Compute bootstrap confidence interval for the mean of `scores`.

    Parameters
    ----------
    scores      : list of per-example 0/1 (or continuous) scores
    n_bootstrap : number of bootstrap resamples
    confidence  : confidence level, e.g. 0.95 for 95% CI

    Returns
    -------
    {
        "mean":      float,
        "ci_lower":  float,
        "ci_upper":  float,
        "std":       float,
        "n":         int,
    }
    """
    arr = np.asarray(scores, dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": float("nan"), "ci_lower": float("nan"),
                "ci_upper": float("nan"), "std": float("nan"), "n": 0}

    rng = np.random.default_rng(seed=42)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        "mean":     round(float(arr.mean()), 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "std":      round(float(arr.std()), 6),
        "n":        n,
    }


# ── Two-sample bootstrap test ─────────────────────────────────────────────────

def compare_models_significance(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 10_000,
) -> dict:
    """
    Two-sided bootstrap permutation test: is model A significantly better than B?

    The null hypothesis is that the two score distributions are identical.
    We permute the labels (A vs B) to build the null distribution of |delta|,
    then compare the observed |delta| against that distribution.

    Returns
    -------
    {
        "mean_a":     float,
        "mean_b":     float,
        "delta":      float,   # mean_a - mean_b
        "p_value":    float,
        "significant": bool,   # p_value < 0.05
        "ci_lower":   float,   # 95% CI on delta
        "ci_upper":   float,
    }
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    observed_delta = float(a.mean() - b.mean())

    # Bootstrap CI on the delta
    rng = np.random.default_rng(seed=42)
    boot_deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        boot_deltas[i] = sa.mean() - sb.mean()

    # Permutation-based p-value (two-sided)
    combined = np.concatenate([a, b])
    perm_deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        perm = rng.permutation(combined)
        perm_deltas[i] = perm[: len(a)].mean() - perm[len(a) :].mean()

    p_value = float(np.mean(np.abs(perm_deltas) >= abs(observed_delta)))

    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))

    return {
        "mean_a":     round(float(a.mean()), 6),
        "mean_b":     round(float(b.mean()), 6),
        "delta":      round(observed_delta, 6),
        "p_value":    round(p_value, 4),
        "significant": p_value < 0.05,
        "ci_lower":   round(ci_lower, 6),
        "ci_upper":   round(ci_upper, 6),
    }


# ── Power analysis ────────────────────────────────────────────────────────────

def power_analysis(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    baseline: float = 0.5,
) -> int:
    """
    Estimate the number of eval examples needed to detect `effect_size`
    difference in accuracy (proportion) with the given power.

    Uses the one-sample proportion formula (margin-of-error approach):

        n = ceil( z_alpha^2 * p*(1-p) / effect_size^2 )

    where z_alpha is the critical value for the two-tailed significance level
    and p is the assumed baseline accuracy.  This formula is standard in survey
    statistics and LLM eval literature (e.g. Eval Harness docs).

    Parameters
    ----------
    effect_size : minimum detectable absolute accuracy difference, e.g. 0.02 for 2 pp
    alpha       : significance level (two-tailed), default 0.05
    power       : (informational) desired power; for a two-sample comparison,
                  multiply the result by ~2 / power
    baseline    : assumed baseline accuracy, default 0.5 (maximises variance)

    Returns
    -------
    int: required number of examples
    """
    if effect_size <= 0:
        raise ValueError("effect_size must be positive")

    z_alpha = _norm_ppf(1.0 - alpha / 2.0)  # two-tailed critical value
    n = (z_alpha ** 2) * baseline * (1.0 - baseline) / (effect_size ** 2)
    return math.ceil(n)


def _norm_ppf(p: float) -> float:
    """Percent-point function (inverse CDF) of standard normal via rational approx."""
    # Abramowitz & Stegun approximation — accurate to ~1e-4
    if p <= 0 or p >= 1:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if p < 0.5:
        return -_norm_ppf(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c = (2.515517, 0.802853, 0.010328)
    d = (1.432788, 0.189269, 0.001308)
    num = c[0] + c[1] * t + c[2] * t * t
    den = 1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t
    return t - num / den


# ── Report generator ──────────────────────────────────────────────────────────

def generate_report(
    results_csv: str = "detailed_results.csv",
    output_path: str = "results/statistical_report.json",
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
) -> None:
    """
    Loads detailed_results.csv, computes bootstrap CIs for each model/metric,
    prints a formatted report, and saves a JSON.

    Expected columns: task_id, model, attempt, passed, error_type, latency, tokens
    """
    csv_p = Path(results_csv)
    if not csv_p.exists():
        print(f"Results file not found: {results_csv}")
        return

    df = pd.read_csv(csv_p)

    # Normalise 'passed' to bool/int
    if "passed" in df.columns:
        df["passed"] = df["passed"].map(
            lambda x: 1 if str(x).strip().lower() in ("true", "1", "yes") else 0
        )

    # Detect per-model score columns
    report: dict = {
        "source": results_csv,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "models": {},
        "pairwise": [],
        "power_analysis": {},
    }

    ci_str = f"{int(confidence * 100)}%"

    # ── Per-model bootstrap CI ────────────────────────────────────────────────
    if "model" in df.columns and "passed" in df.columns:
        print(f"\n{'Metric':<35} {'Score':>7}  {ci_str + ' CI':<20}  {'N':>5}")
        print("-" * 72)

        model_stats: dict[str, dict] = {}
        for model_name, grp in df.groupby("model"):
            scores = grp["passed"].tolist()
            ci = bootstrap_ci(scores, n_bootstrap=n_bootstrap, confidence=confidence)
            model_stats[str(model_name)] = ci
            score_pct  = f"{ci['mean']:.1%}"
            lower_pct  = f"{ci['ci_lower']:.1%}"
            upper_pct  = f"{ci['ci_upper']:.1%}"
            ci_range   = f"[{lower_pct}, {upper_pct}]"
            print(f"  {str(model_name):<33} {score_pct:>7}  {ci_range:<20}  {ci['n']:>5}")

        report["models"] = model_stats

        # ── Pairwise significance ─────────────────────────────────────────────
        model_names = sorted(model_stats.keys())
        if len(model_names) >= 2:
            print(f"\n{'Comparison':<60}  {'delta':>7}  {'p-value':>8}  Sig?")
            print("-" * 85)
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    ma, mb = model_names[i], model_names[j]
                    scores_a = df[df["model"] == ma]["passed"].tolist()
                    scores_b = df[df["model"] == mb]["passed"].tolist()
                    pw = compare_models_significance(scores_a, scores_b, n_bootstrap=n_bootstrap)
                    pw["model_a"] = ma
                    pw["model_b"] = mb
                    report["pairwise"].append(pw)
                    sig_flag = "*" if pw["significant"] else ""
                    cmp_label = f"{ma}  vs  {mb}"
                    print(
                        f"  {cmp_label:<58}  {pw['delta']:>+7.3f}  {pw['p_value']:>8.4f}  {sig_flag}"
                    )
    else:
        # Fallback: treat the entire 'passed' column as one metric
        if "passed" in df.columns:
            scores = df["passed"].tolist()
            ci = bootstrap_ci(scores, n_bootstrap=n_bootstrap, confidence=confidence)
            report["models"]["all"] = ci
            print(f"\nAll rows combined:")
            print(f"  Score: {ci['mean']:.3f}  {ci_str} CI: [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]  N={ci['n']}")

    # ── Power analysis table ──────────────────────────────────────────────────
    print(f"\n{'Effect size':>12}  {'Required N':>12}")
    print("-" * 28)
    pa: dict[str, int] = {}
    for es in (0.01, 0.02, 0.03, 0.05, 0.10):
        n_req = power_analysis(es, alpha=0.05, power=0.80)
        pa[str(es)] = n_req
        print(f"  {es:>10.0%}  {n_req:>12,}")

    report["power_analysis"] = pa

    # ── Save JSON ─────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved statistical report → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals and significance tests for benchmark scores."
    )
    parser.add_argument(
        "--results", default="detailed_results.csv",
        help="CSV with per-example results (needs 'passed' column)",
    )
    parser.add_argument(
        "--output", default="results/statistical_report.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--n_bootstrap", type=int, default=10_000,
        help="Number of bootstrap resamples (default: 10000)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.95,
        help="Confidence level (default: 0.95)",
    )
    parser.add_argument(
        "--power", action="store_true",
        help="Run a standalone power analysis table and exit",
    )
    parser.add_argument(
        "--effect_size", type=float, default=0.02,
        help="Effect size for standalone power analysis (default: 0.02)",
    )
    args = parser.parse_args()

    if args.power:
        n_req = power_analysis(args.effect_size, alpha=0.05, power=0.80)
        print(
            f"To detect a {args.effect_size:.1%} accuracy difference "
            f"with 80% power (alpha=0.05), you need ~{n_req:,} examples."
        )
        return

    generate_report(
        results_csv=args.results,
        output_path=args.output,
        n_bootstrap=args.n_bootstrap,
        confidence=args.confidence,
    )


if __name__ == "__main__":
    main()
