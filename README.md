# Code Agent Eval Benchmark

A lightweight benchmark for evaluating code-oriented language models on task performance, latency, and token efficiency.

This project compares multiple open models on a code agent style evaluation and visualizes the tradeoffs between quality and speed.

---

## Overview

The goal of this benchmark is simple:

- Measure **functional correctness** using Pass@1  
- Track **average latency per task**
- Record **token usage**
- Make tradeoffs visible

The current comparison includes:

- `meta-llama/llama-4-scout-17b-16e-instruct`
- `llama-3.1-8b-instant`
- `allam-2-7b`
- `openai/gpt-oss-20b`

The results are summarized both in CSV format and in a visual performance vs latency chart.

---

Files:
- **eval_benchmark.ipynb**  
  Main evaluation notebook. Runs model calls, captures latency and tokens, and computes summary metrics.

- **detailed_results.csv**  
  Row-level results for each task and model.

- **metrics_summary.csv**  
  Aggregated metrics per model.

- **performance_vs_latency.png**  
  Bubble chart showing quality vs speed tradeoffs.

---

## Metrics

### Pass@1 (%)

Percentage of tasks solved correctly on the first attempt.  
Higher means better reliability without retries.

### Average Latency (seconds)

Mean time per task.  
Lower is better for interactive or agent-based systems.

### Average Tokens

Average number of tokens consumed per task.  
Useful for estimating cost and efficiency.

---

## Current Results

| Model | Pass@1 | Avg Latency (s) | Avg Tokens |
|-------|--------|-----------------|------------|
| meta-llama/llama-4-scout-17b-16e-instruct | 100% | ~0.48 | ~305 |
| llama-3.1-8b-instant | 50% | ~0.13 | ~215 |
| allam-2-7b | 10% | ~0.34 | ~280 |
| openai/gpt-oss-20b | 10% | ~0.41 | ~490 |

### Qwen2.5-7B-Instruct — Multi-Axis Benchmark (measured on NVIDIA A30)

| Benchmark | Metric | Score |
|-----------|--------|-------|
| GSM8K (n=50) | Accuracy | **0.540** |
| HumanEval-style (n=10) | pass@1 | **0.700** |
| LLM-as-Judge (n=10) | Avg score | **8.2/10** |
| &nbsp;&nbsp;↳ Math reasoning | score | 8.5/10 |
| &nbsp;&nbsp;↳ Code generation | score | 8.0/10 |
| &nbsp;&nbsp;↳ Logical reasoning | score | 9.0/10 |
| &nbsp;&nbsp;↳ Instruction following | score | 8.0/10 |
| &nbsp;&nbsp;↳ Knowledge | score | 7.5/10 |

Full results: `results/base_model/eval_results.json`

### Quick Takeaways

- `llama-3.1-8b-instant` is the fastest model in this benchmark.
- `meta-llama/llama-4-scout-17b-16e-instruct` achieved the highest Pass@1 score.
- `openai/gpt-oss-20b` consumed the most tokens in this run.
- Smaller models tend to be faster, but accuracy varies significantly depending on task complexity.

---

## Visualization

The `performance_vs_latency.png` file plots:

- **X-axis:** Average latency  
- **Y-axis:** Pass@1  
- **Bubble size:** Average tokens  

This makes it easy to compare speed, quality, and efficiency at a glance.

---

## Multi-Step Tool-Use Evaluation (`eval_tool_use.py`)

A ReAct-style harness that evaluates agents on 14 tasks requiring multi-turn tool calls. The agent thinks, calls a tool, receives an `[OBS]` result, and iterates until it emits `<final_answer>`.

### Tools

| Tool | Description |
|------|-------------|
| `python_exec` | Subprocess Python sandbox — arbitrary computation |
| `calculator` | Safe expression evaluator (no arbitrary code) |
| `lookup` | Retrieves constants from a knowledge base (π, speed of light, etc.) |

### Task categories (14 total)

| Category | Tasks | What it tests |
|----------|-------|---------------|
| `computation` | 5 | Single tool call for direct calculation |
| `multi_step` | 5 | Chained reasoning: digit sum of factorial, Collatz, geometric series |
| `knowledge_compute` | 2 | Lookup → calculator: circle area with exact π, light travel time |
| `error_recovery` | 2 | First call fails (sqrt(-1), div-by-0) → agent retries correct call |

### Rule-based ceiling agent (demo mode, no GPU)

```bash
python eval_tool_use.py --mode demo
```

**Measured results** (deterministic ceiling, 14 tasks):

| Metric | Value |
|--------|-------|
| Task success | **100.0%** (14/14) |
| Avg steps/task | **1.3** |
| Tool accuracy | **88.9%** (expected errors counted) |
| Error recovery | **100.0%** (2/2 error tasks recovered) |
| Avg latency | **0.015 s** (subprocess only) |

Per-category: computation 100% · multi_step 100% · knowledge_compute 100% · error_recovery 100%

### Real model evaluation

```bash
# Requires GPU + Qwen2.5-7B-Instruct (or any HF causal LM)
python eval_tool_use.py --mode hf --model Qwen/Qwen2.5-7B-Instruct
```

Results are written to `results/tool_use_results.json`.

### Extending the harness

Add a task by appending to `TASKS`:
```python
Task("my_task", "Compute the 20th prime number.", "73", category="computation")
```

Add a tool by registering in `TOOLS` and adding a description to `TOOL_DESCRIPTIONS`.

---

## Extended Evaluation: GSM8K + HumanEval + LLM-as-Judge (`eval_benchmarks.py`)

A new multi-axis harness (`eval_benchmarks.py`) extends the original latency/pass@1 benchmark with:

### Benchmarks

| Benchmark | Task | Metric | How |
|-----------|------|--------|-----|
| **GSM8K** | Math reasoning (7,473 problems) | Exact-match accuracy | Numeric extraction + ±1e-4 tolerance |
| **HumanEval-style** | Code generation (10 representative problems) | pass@1 | Subprocess execution + test assertion |
| **LLM-as-Judge** | Open-ended quality (10 questions) | Score 0–10 | Self-judge via structured prompt |

### HumanEval-style categories covered

- List manipulation (`has_close_elements`, `intersperse`, `rolling_max`)
- String parsing (`separate_paren_groups`, `parse_nested_parens`, `filter_by_substring`)
- Numeric (`truncate_number`, `below_zero`, `mean_absolute_deviation`, `sum_product`)

### LLM-as-Judge categories

- Math reasoning, Code generation, Logical reasoning, Instruction following, Knowledge

### Usage

```bash
pip install torch transformers peft datasets

# Evaluate base model
python eval_benchmarks.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --benchmarks gsm8k humaneval llm_judge \
    --gsm8k_n 100 --humaneval_n 10 --judge_n 10 \
    --output_dir results/base

# Evaluate fine-tuned model (with LoRA adapter)
python eval_benchmarks.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --adapter_path /path/to/lora/adapter \
    --benchmarks gsm8k humaneval llm_judge \
    --output_dir results/finetuned
```

Outputs: `results/eval_results.json` + `results/eval_report.md`

---

## How to Run

### 1. Install Dependencies

```bash
pip install pandas matplotlib
# You may also need SDKs or API clients for the models you are evaluating.
```

### 2. Run the Notebook

```bash
jupyter notebook eval_benchmark.ipynb
```

The notebook will load tasks, query each model, measure latency and token usage, compute Pass@1, and export `detailed_results.csv`, `metrics_summary.csv`, and `performance_vs_latency.png`.

### Adding a New Model

Add its inference call in `eval_benchmark.ipynb`, capture model output, latency, and token usage, re-run, and regenerate the summary CSV and visualization. Keep logging consistent so comparisons remain fair.

---

## Why This Benchmark Exists

When building code agents or automation systems, accuracy alone is not enough. You also care about response time, token efficiency, cost, and reliability under tool-call failures. This project provides a simple, reproducible way to compare those tradeoffs — including a ReAct agent harness that tests error recovery, multi-step chaining, and tool accuracy, not just single-turn pass@1.

