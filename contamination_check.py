"""
Checks for benchmark contamination: n-gram overlap between eval questions
and a reference training corpus (UltraFeedback or a sample thereof).

A question is flagged as potentially contaminated if more than `threshold`
fraction of its character-level n-grams appear in the corpus.

Usage:
    python contamination_check.py --eval_file detailed_results.csv --threshold 0.2
    python contamination_check.py --eval_file detailed_results.csv --corpus openai/gsm8k --corpus_split train
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ── N-gram helpers ────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _char_ngrams(text: str, n: int) -> list[str]:
    t = _normalise(text)
    if len(t) < n:
        return [t]
    return [t[i : i + n] for i in range(len(t) - n + 1)]


def build_ngram_index(texts: list[str], n: int = 13) -> set[str]:
    """Build the set of all n-grams present in the corpus."""
    index: set[str] = set()
    for text in texts:
        index.update(_char_ngrams(text, n))
    return index


# ── Contamination scoring ─────────────────────────────────────────────────────

def check_contamination(
    eval_questions: list[str],
    corpus_ngrams: set[str],
    n: int = 13,
    threshold: float = 0.2,
) -> dict[int, dict]:
    """
    For each eval question, compute the fraction of its n-grams that appear
    in the corpus.  Flag as contaminated if fraction > threshold.

    Returns:
        {
            question_idx: {
                "question_preview": str,
                "n_question_ngrams": int,
                "n_overlap": int,
                "overlap_frac": float,
                "contaminated": bool,
            }
        }
    """
    results: dict[int, dict] = {}
    for idx, question in enumerate(eval_questions):
        qgrams = _char_ngrams(question, n)
        n_total = len(qgrams)
        if n_total == 0:
            results[idx] = {
                "question_preview": question[:80],
                "n_question_ngrams": 0,
                "n_overlap": 0,
                "overlap_frac": 0.0,
                "contaminated": False,
            }
            continue
        n_overlap = sum(1 for g in qgrams if g in corpus_ngrams)
        frac = n_overlap / n_total
        results[idx] = {
            "question_preview": question[:80],
            "n_question_ngrams": n_total,
            "n_overlap": n_overlap,
            "overlap_frac": round(frac, 4),
            "contaminated": frac > threshold,
        }
    return results


# ── Full pipeline ─────────────────────────────────────────────────────────────

def _load_eval_questions(eval_file: str) -> list[str]:
    """
    Load questions from the eval file.

    Handles two formats:
    1. detailed_results.csv — extracts unique task IDs as question proxies
       (actual question text is not stored; task_id is used as a stand-in)
    2. Any CSV with a 'question' or 'prompt' column
    """
    df = pd.read_csv(eval_file)
    for col in ("question", "prompt", "task", "task_id"):
        if col in df.columns:
            return df[col].dropna().unique().tolist()
    # Fallback: use all string columns
    return df.iloc[:, 0].dropna().unique().tolist()


def _load_corpus_texts(
    corpus_name: str,
    corpus_split: str,
    corpus_sample: int,
) -> list[str]:
    """
    Load a sample of texts from a HuggingFace dataset.
    Falls back gracefully if the dataset is unavailable.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets library not installed. pip install datasets")
        return []

    try:
        print(f"Loading corpus: {corpus_name} (split={corpus_split}, sample={corpus_sample})")
        ds = load_dataset(
            corpus_name,
            split=corpus_split,
            streaming=True,
            trust_remote_code=False,
        )
    except Exception as e:
        print(f"Could not load dataset {corpus_name!r}: {e}")
        return []

    texts: list[str] = []
    text_cols = ("prompt", "chosen", "rejected", "instruction", "input", "text", "question")

    for i, row in enumerate(ds):
        if i >= corpus_sample:
            break
        for col in text_cols:
            val = row.get(col)
            if isinstance(val, str) and val.strip():
                texts.append(val)
                break
            elif isinstance(val, list):
                # Chosen/rejected in ultrafeedback are lists of dicts
                for turn in val:
                    if isinstance(turn, dict) and "content" in turn:
                        texts.append(turn["content"])
                break

    print(f"Loaded {len(texts)} corpus texts.")
    return texts


def run_contamination_check(
    eval_file: str = "detailed_results.csv",
    corpus_name: str = "HuggingFaceH4/ultrafeedback_binarized",
    corpus_split: str = "train_prefs",
    corpus_sample: int = 5000,
    n: int = 13,
    threshold: float = 0.2,
    output_path: str = "results/contamination_report.json",
) -> dict:
    """
    Full contamination check pipeline.

    1. Load eval questions from eval_file.
    2. Load a sample of the training corpus.
    3. Build n-gram index of the corpus.
    4. Score each eval question for overlap.
    5. Print and save a report.
    """
    print(f"=== Contamination Check ===")
    print(f"  Eval file:    {eval_file}")
    print(f"  Corpus:       {corpus_name}")
    print(f"  N-gram size:  {n}")
    print(f"  Threshold:    {threshold}\n")

    eval_questions = _load_eval_questions(eval_file)
    print(f"Loaded {len(eval_questions)} unique eval questions/IDs.\n")

    corpus_texts = _load_corpus_texts(corpus_name, corpus_split, corpus_sample)

    if not corpus_texts:
        print("No corpus texts loaded — contamination check skipped.")
        return {}

    print(f"Building {n}-gram index over {len(corpus_texts)} corpus texts...")
    corpus_ngrams = build_ngram_index(corpus_texts, n=n)
    print(f"Index size: {len(corpus_ngrams):,} unique {n}-grams.\n")

    results = check_contamination(eval_questions, corpus_ngrams, n=n, threshold=threshold)

    # Summarise
    n_total = len(results)
    n_contaminated = sum(1 for r in results.values() if r["contaminated"])
    avg_overlap = (
        sum(r["overlap_frac"] for r in results.values()) / n_total if n_total > 0 else 0.0
    )

    report = {
        "eval_file": eval_file,
        "corpus": corpus_name,
        "ngram_size": n,
        "threshold": threshold,
        "n_questions": n_total,
        "n_contaminated": n_contaminated,
        "contamination_rate": round(n_contaminated / max(n_total, 1), 4),
        "avg_overlap_frac": round(avg_overlap, 4),
        "details": results,
    }

    print(f"=== Results ===")
    print(f"  Questions checked:    {n_total}")
    print(f"  Contaminated (>{threshold:.0%}): {n_contaminated}  ({report['contamination_rate']:.1%})")
    print(f"  Avg overlap fraction: {avg_overlap:.4f}")

    if n_contaminated > 0:
        print("\n  Flagged questions:")
        for idx, r in results.items():
            if r["contaminated"]:
                print(
                    f"    [{idx}] overlap={r['overlap_frac']:.3f}  "
                    f"preview={r['question_preview']!r}"
                )
    else:
        print("\n  No questions flagged — eval set appears clean.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Convert int keys to str for JSON serialisation
    report["details"] = {str(k): v for k, v in report["details"].items()}
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report → {output_path}")

    return report


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Check for n-gram contamination between eval questions and a training corpus."
    )
    parser.add_argument(
        "--eval_file", default="detailed_results.csv",
        help="CSV file with eval questions (columns: question/prompt/task_id)",
    )
    parser.add_argument(
        "--corpus", default="HuggingFaceH4/ultrafeedback_binarized",
        dest="corpus_name",
        help="HuggingFace dataset name for the training corpus",
    )
    parser.add_argument(
        "--corpus_split", default="train_prefs",
        help="Dataset split to use (default: train_prefs)",
    )
    parser.add_argument(
        "--corpus_sample", type=int, default=5000,
        help="Number of corpus examples to sample (default: 5000)",
    )
    parser.add_argument(
        "--ngram_size", type=int, default=13,
        help="Character n-gram size (default: 13, per Gopher/Chinchilla practice)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2,
        help="Overlap fraction threshold above which a question is flagged (default: 0.2)",
    )
    parser.add_argument(
        "--output", default="results/contamination_report.json",
        dest="output_path",
        help="Path to save the JSON report",
    )
    args = parser.parse_args()

    run_contamination_check(
        eval_file=args.eval_file,
        corpus_name=args.corpus_name,
        corpus_split=args.corpus_split,
        corpus_sample=args.corpus_sample,
        n=args.ngram_size,
        threshold=args.threshold,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
