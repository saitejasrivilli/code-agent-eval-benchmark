"""
eval_multiturn.py

Multi-turn conversation evaluator for agent post-training.

Unlike eval_tool_use.py (single ReAct chains), this harness runs 3-5 turn
conversations where each turn depends on the previous — testing whether
the model maintains context, follows the original instructions, and does
not drift from the task.

Three failure modes measured:
  context_loss      — model ignores or contradicts a prior turn's result
  instruction_drift — model stops following the original instruction by turn 4-5
  error_propagation — an early mistake compounds in later turns vs. recovers

20 multi-turn tasks across 4 categories:
  reference_chain   (5) : "compute X" → "multiply that by Y" → "what was my first question?"
  state_tracking    (5) : maintain a running list/dict across turns
  instruction_stack (5) : new instruction per turn, all must hold simultaneously by final turn
  recovery_test     (5) : inject a deliberate error in turn 2, measure whether turn 3 recovers

Usage:
    python eval_multiturn.py --mode demo    # no GPU, deterministic
    CUDA_VISIBLE_DEVICES=0 python eval_multiturn.py --mode hf \\
        --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Turn:
    user: str
    expected: str
    depends_on: Optional[int] = None   # index of prior turn whose result this references
    eval_fn: str = "contains"          # exact_match | numeric_match | contains | json_valid


@dataclass
class MultiTurnTask:
    task_id: str
    category: str
    turns: List[Turn]
    description: str


# ── Eval functions ─────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    return text.lower().strip()


def _best_numeric(response: str, expected: str) -> Optional[Tuple[float, float]]:
    """
    Extract the best candidate number from response to compare against expected.

    Strategy (in order):
    1. If response contains '=', take the number immediately after the last '='.
    2. Among all numbers in the response, pick the one closest to the target.
    3. Fall back to the last number.

    This handles:
    - "15 × 17 = 255"              → picks 255
    - "There are 200 cm in 2 m."   → picks 200 (closest to expected 200)
    """
    nums_e = re.findall(r"-?\d+(?:\.\d+)?", expected)
    if not nums_e:
        return None
    try:
        target = float(nums_e[0])
    except ValueError:
        return None

    # Strategy 1: number after last '='
    if "=" in response:
        after_eq = response.rsplit("=", 1)[-1]
        nums_after = re.findall(r"-?\d+(?:\.\d+)?", after_eq)
        if nums_after:
            try:
                candidate = float(nums_after[0])
                if abs(candidate - target) < 1.0:
                    return candidate, target
            except ValueError:
                pass

    nums_r = re.findall(r"-?\d+(?:\.\d+)?", response)
    if not nums_r:
        return None

    # Strategy 2: closest number to target
    try:
        floats = [float(n) for n in nums_r]
        best = min(floats, key=lambda x: abs(x - target))
        return best, target
    except ValueError:
        return None


def _eval_exact_match(response: str, expected: str) -> bool:
    pair = _best_numeric(response, expected)
    if pair is not None:
        return abs(pair[0] - pair[1]) < 0.02
    return _normalise(response) == _normalise(expected)


def _eval_numeric_match(response: str, expected: str) -> bool:
    pair = _best_numeric(response, expected)
    if pair is None:
        return False
    return abs(pair[0] - pair[1]) < 0.02


def _eval_contains(response: str, expected: str) -> bool:
    """
    Check whether the response contains the expected substring(s).

    Separator semantics:
      '|'  — OR  : any one of the alternatives must appear  (e.g. acknowledgement variants)
      '&'  — AND : every term must appear simultaneously    (e.g. multi-part answers)

    If neither separator is present the whole string is checked as-is.
    """
    resp_lower = response.lower()
    if "&" in expected:
        parts = [p.strip() for p in expected.split("&")]
        return all(p.lower() in resp_lower for p in parts)
    if "|" in expected:
        parts = [p.strip() for p in expected.split("|")]
        return any(p.lower() in resp_lower for p in parts)
    return expected.lower() in resp_lower


def _eval_json_valid(response: str, _expected: str) -> bool:
    # Look for any JSON object/array in the response
    for m in re.finditer(r"[\[{].*?[\]}]", response, re.DOTALL):
        try:
            json.loads(m.group())
            return True
        except json.JSONDecodeError:
            pass
    return False


EVAL_FNS: Dict[str, Callable[[str, str], bool]] = {
    "exact_match":   _eval_exact_match,
    "numeric_match": _eval_numeric_match,
    "contains":      _eval_contains,
    "json_valid":    _eval_json_valid,
}


def score_turn(response: str, turn: Turn) -> bool:
    fn = EVAL_FNS.get(turn.eval_fn, _eval_contains)
    return fn(response, turn.expected)


# ── Task definitions ──────────────────────────────────────────────────────────

TASKS: List[MultiTurnTask] = [

    # ── reference_chain ──────────────────────────────────────────────────────

    MultiTurnTask(
        task_id="rc_01_arithmetic_chain",
        category="reference_chain",
        description="Basic arithmetic chain: product → add → divide → recall first result",
        turns=[
            Turn(
                user="What is 15 × 17?",
                expected="255",
                eval_fn="numeric_match",
            ),
            Turn(
                user="Add 45 to that result.",
                expected="300",
                depends_on=0,
                eval_fn="numeric_match",
            ),
            Turn(
                user="Divide the current number by 3.",
                expected="100",
                depends_on=1,
                eval_fn="numeric_match",
            ),
            Turn(
                user="What was the original product you computed in the very first step?",
                expected="255",
                depends_on=0,
                eval_fn="numeric_match",
            ),
        ],
    ),

    MultiTurnTask(
        task_id="rc_02_primes_and_sum",
        category="reference_chain",
        description="List primes → sum them → primality check → next prime",
        turns=[
            Turn(
                user="List the first 5 prime numbers.",
                expected="2&3&5&7&11",
                eval_fn="contains",
            ),
            Turn(
                user="Sum all the numbers you just listed.",
                expected="28",
                depends_on=0,
                eval_fn="numeric_match",
            ),
            Turn(
                user="Is 28 a prime number?",
                expected="no",
                depends_on=1,
                eval_fn="contains",
            ),
            Turn(
                user="What is the next prime number after 11?",
                expected="13",
                eval_fn="numeric_match",
            ),
        ],
    ),

    MultiTurnTask(
        task_id="rc_03_string_transform_chain",
        category="reference_chain",
        description="Reverse a word → count letters → multiply by prior result → recall word",
        turns=[
            Turn(
                user='Reverse the word "python" and tell me the result.',
                expected="nohtyp",
                eval_fn="contains",
            ),
            Turn(
                user="How many letters does that reversed word have?",
                expected="6",
                depends_on=0,
                eval_fn="numeric_match",
            ),
            Turn(
                user="Multiply that letter count by 7.",
                expected="42",
                depends_on=1,
                eval_fn="numeric_match",
            ),
            Turn(
                user="What was the original word I gave you to reverse?",
                expected="python",
                depends_on=0,
                eval_fn="contains",
            ),
        ],
    ),

    MultiTurnTask(
        task_id="rc_04_unit_conversion_chain",
        category="reference_chain",
        description="Convert km → miles → feet → recall original km",
        turns=[
            Turn(
                user="Convert 10 kilometres to miles. (Use 1 km = 0.621371 miles.)",
                expected="6.21",
                eval_fn="numeric_match",
            ),
            Turn(
                user="Convert that miles figure to feet. (Use 1 mile = 5280 feet.)",
                expected="32808",
                depends_on=0,
                eval_fn="contains",
            ),
            Turn(
                user="Round that feet value to the nearest thousand.",
                expected="33000",
                depends_on=1,
                eval_fn="numeric_match",
            ),
            Turn(
                user="What was the original distance in kilometres that I asked you to convert?",
                expected="10",
                depends_on=0,
                eval_fn="numeric_match",
            ),
        ],
    ),

    MultiTurnTask(
        task_id="rc_05_factorial_chain",
        category="reference_chain",
        description="Compute factorial → digit sum → square it → recall factorial",
        turns=[
            Turn(
                user="What is 7 factorial (7!)?",
                expected="5040",
                eval_fn="numeric_match",
            ),
            Turn(
                user="What is the sum of the individual digits of that number?",
                expected="9",
                depends_on=0,
                eval_fn="numeric_match",
            ),
            Turn(
                user="Square that digit sum.",
                expected="81",
                depends_on=1,
                eval_fn="numeric_match",
            ),
            Turn(
                user="What was the value of 7! that you computed at the start of our conversation?",
                expected="5040",
                depends_on=0,
                eval_fn="numeric_match",
            ),
        ],
    ),

    # ── state_tracking ────────────────────────────────────────────────────────

    MultiTurnTask(
        task_id="st_01_shopping_cart",
        category="state_tracking",
        description="Maintain a shopping cart across add/remove operations",
        turns=[
            Turn(
                user="Start a shopping cart. Add: apples (3 units), milk (2 units), bread (1 unit).",
                expected="apples&milk&bread",
                eval_fn="contains",
            ),
            Turn(
                user="Remove apples from the cart and add eggs (6 units).",
                expected="milk&bread&eggs",
                eval_fn="contains",
            ),
            Turn(
                user="What items are currently in my cart?",
                expected="milk&bread&eggs",
                depends_on=1,
                eval_fn="contains",
            ),
            Turn(
                user="How many total units are in the cart right now?",
                expected="9",
                depends_on=1,
                eval_fn="numeric_match",
            ),
        ],
    ),

    MultiTurnTask(
        task_id="st_02_score_tracker",
        category="state_tracking",
        description="Track scores for two players across multiple rounds",
        turns=[
            Turn(
                user="Track scores for Alice and Bob. Round 1: Alice scores 10, Bob scores 7.",
                expected="Alice&Bob",
                eval_fn="contains",
            ),
            Turn(
                user="Round 2: Alice scores 5, Bob scores 12.",
                expected="Alice&Bob",
                eval_fn="contains",
            ),
            Turn(
                user="Round 3: Alice scores 8, Bob scores 3.",
                expected="Alice&23&Bob&22",
                eval_fn="contains",
            ),
            Turn(
                user="Who is currently winning and by how many points?",
                expected="Alice&1",
                depends_on=2,
                eval_fn="contains",
            ),
        ],
    ),

    MultiTurnTask(
        task_id="st_03_word_list",
        category="state_tracking",
        description="Build and filter a list of words across turns",
        turns=[
            Turn(
                user="Start a word list. Add: elephant, apple, banana, cherry.",
                expected="elephant&apple&banana&cherry",
                eval_fn="contains",
            ),
            Turn(
                user="Remove any words that start with a vowel.",
                expected="banana&cherry",
                eval_fn="contains",
            ),
            Turn(
                user="Add 'mango' and 'orange' to the list.",
                expected="banana&cherry&mango&orange",
                eval_fn="contains",
            ),
            Turn(
                user="How many words are in the list now?",
                expected="4",
                depends_on=2,
                eval_fn="numeric_match",
            ),
        ],
    ),

    MultiTurnTask(
        task_id="st_04_inventory",
        category="state_tracking",
        description="Warehouse inventory management across several transactions",
        turns=[
            Turn(
                user="Initialize warehouse inventory: widgets=100, gadgets=50, gizmos=75.",
                expected="widgets&gadgets&gizmos",
                eval_fn="contains",
            ),
            Turn(
                user="Process shipment: receive 30 more widgets and ship out 20 gadgets.",
                expected="widgets&130&gadgets&30",
                eval_fn="contains",
            ),
            Turn(
                user="Ship out 25 gizmos and receive 15 gadgets.",
                expected="gadgets&45&gizmos&50",
                eval_fn="contains",
            ),
            Turn(
                user="What is the total item count across all product types?",
                expected="225",
                depends_on=2,
                eval_fn="numeric_match",
            ),
        ],
    ),

    MultiTurnTask(
        task_id="st_05_todo_list",
        category="state_tracking",
        description="Manage a to-do list with add/complete/remove operations",
        turns=[
            Turn(
                user="Create a to-do list. Add tasks: 'write report', 'send email', 'schedule meeting', 'review code'.",
                expected="write report&send email&schedule meeting&review code",
                eval_fn="contains",
            ),
            Turn(
                user="Mark 'send email' and 'review code' as completed. Remove them from the list.",
                expected="write report&schedule meeting",
                eval_fn="contains",
            ),
            Turn(
                user="Add a new task: 'prepare slides'. What tasks remain on the list?",
                expected="write report&schedule meeting&prepare slides",
                depends_on=1,
                eval_fn="contains",
            ),
            Turn(
                user="How many tasks are still on the list?",
                expected="3",
                depends_on=2,
                eval_fn="numeric_match",
            ),
        ],
    ),

    # ── instruction_stack ─────────────────────────────────────────────────────

    MultiTurnTask(
        task_id="is_01_sentence_format",
        category="instruction_stack",
        description="Accumulate formatting constraints: 3-sentence cap, capitals, 'Done.' suffix",
        turns=[
            Turn(
                user="From now on, answer all my questions in exactly 3 sentences.",
                expected="understood|will|3 sentences|okay|sure|got it",
                eval_fn="contains",
            ),
            Turn(
                user="Also, make sure every sentence starts with a capital letter.",
                expected="understood|will|okay|sure|got it",
                eval_fn="contains",
            ),
            Turn(
                user="What is photosynthesis?",
                # Check: response must contain some content about photosynthesis
                # The harness checks 3-sentence and capital constraints separately via _check_instruction_stack
                expected="photosynthesis|light|plant|chloro",
                eval_fn="contains",
            ),
            Turn(
                user="Also, end every response with exactly the word 'Done.'",
                expected="understood|will|okay|sure|got it",
                eval_fn="contains",
            ),
            Turn(
                user="What is osmosis?",
                expected="osmosis&Done.",
                depends_on=0,
                eval_fn="contains",
            ),
        ],
    ),

    MultiTurnTask(
        task_id="is_02_list_format",
        category="instruction_stack",
        description="Accumulate: bullet points, uppercase bullets, numbered prefix",
        turns=[
            Turn(
                user="Please always format your answers as a bulleted list using hyphens (-).",
                expected="understood|will|okay|sure|got it",
                eval_fn="contains",
            ),
            Turn(
                user="Also, write each bullet item in ALL CAPS.",
                expected="understood|will|okay|sure|got it",
                eval_fn="contains",
            ),
            Turn(
                user="Name three programming languages.",
                expected="-",
                eval_fn="contains",
            ),
            Turn(
                user="Also, prefix each bullet with its number, like '1. - ITEM'.",
                expected="understood|will|okay|sure|got it",
                eval_fn="contains",
            ),
            Turn(
                user="Name three databases.",
                expected="1|2|3",
                depends_on=0,
                eval_fn="contains",
            ),
        ],
    ),

    MultiTurnTask(
        task_id="is_03_word_limit",
        category="instruction_stack",
        description="Accumulate: max 20 words, no adjectives, end with question",
        turns=[
            Turn(
                user="Keep all your answers to a maximum of 20 words from now on.",
                expected="understood|will|okay|sure|got it",
                eval_fn="contains",
            ),
            Turn(
                user="Also, do not use any adjectives in your answers.",
                expected="understood|will|okay|sure|got it",
                eval_fn="contains",
            ),
            Turn(
                user="What is gravity?",
                expected="gravity|force|mass|attraction",
                eval_fn="contains",
            ),
            Turn(
                user="Also, end every answer with a question mark (?).",
                expected="understood|will|okay|sure|got it",
                eval_fn="contains",
            ),
            Turn(
                user="What is electricity?",
                expected="?",
                depends_on=0,
                eval_fn="contains",
            ),
        ],
    ),

    MultiTurnTask(
        task_id="is_04_language_constraints",
        category="instruction_stack",
        description="Accumulate: simple words, no jargon, first-person, single paragraph",
        turns=[
            Turn(
                user="Please use only simple, everyday words — no technical jargon.",
                expected="understood|will|okay|sure|got it",
                eval_fn="contains",
            ),
            Turn(
                user="Also, write in first person (use 'I' and 'we').",
                expected="understood|will|okay|sure|got it",
                eval_fn="contains",
            ),
            Turn(
                user="Explain what a computer does.",
                expected="computer|work|help|do|I",
                eval_fn="contains",
            ),
            Turn(
                user="Also, keep your answer to a single paragraph with no line breaks.",
                expected="understood|will|okay|sure|got it",
                eval_fn="contains",
            ),
            Turn(
                user="Explain what the internet is.",
                expected="internet|connect|I",
                depends_on=0,
                eval_fn="contains",
            ),
        ],
    ),

    MultiTurnTask(
        task_id="is_05_output_format",
        category="instruction_stack",
        description="Accumulate: start with 'Answer:', no filler phrases, end with confidence",
        turns=[
            Turn(
                user="Start every answer with the word 'Answer:' on its own line.",
                expected="understood|will|okay|sure|got it",
                eval_fn="contains",
            ),
            Turn(
                user="Also, never use filler phrases like 'certainly', 'of course', or 'sure'.",
                expected="understood|noted|will do|okay|acknowledged",
                eval_fn="contains",
            ),
            Turn(
                user="What is the capital of France?",
                expected="Answer:&Paris",
                eval_fn="contains",
            ),
            Turn(
                user="Also, end every answer with your confidence level as a percentage.",
                expected="understood|will|okay|acknowledged",
                eval_fn="contains",
            ),
            Turn(
                user="What is the capital of Japan?",
                expected="Answer:&Tokyo&%",
                depends_on=0,
                eval_fn="contains",
            ),
        ],
    ),

    # ── recovery_test ─────────────────────────────────────────────────────────

    MultiTurnTask(
        task_id="rt_01_sqrt_correction",
        category="recovery_test",
        description="Injected wrong sqrt answer; model should self-correct when challenged",
        turns=[
            Turn(
                user="What is the square root of 144?",
                expected="12",
                eval_fn="numeric_match",
            ),
            # Turn index 1 is the injected wrong assistant answer — handled specially
            # by the harness: conversation history has assistant say "13" here.
            Turn(
                user="Are you sure? Please double-check your answer.",
                expected="12",
                depends_on=0,
                eval_fn="numeric_match",
            ),
            Turn(
                user="What was the original number I asked you to take the square root of?",
                expected="144",
                depends_on=0,
                eval_fn="numeric_match",
            ),
        ],
    ),

    MultiTurnTask(
        task_id="rt_02_capital_correction",
        category="recovery_test",
        description="Injected wrong capital city; model should self-correct when challenged",
        turns=[
            Turn(
                user="What is the capital of Australia?",
                expected="Canberra",
                eval_fn="contains",
            ),
            # Injected wrong answer: "Sydney"
            Turn(
                user="I think that might be wrong — can you verify the capital of Australia?",
                expected="Canberra",
                depends_on=0,
                eval_fn="contains",
            ),
            Turn(
                user="What country were we discussing?",
                expected="Australia",
                depends_on=0,
                eval_fn="contains",
            ),
        ],
    ),

    MultiTurnTask(
        task_id="rt_03_formula_correction",
        category="recovery_test",
        description="Injected wrong area formula result; model corrects and continues",
        turns=[
            Turn(
                user="What is the area of a circle with radius 5? Use π ≈ 3.14159.",
                expected="78.54",
                eval_fn="numeric_match",
            ),
            # Injected wrong answer: "area is 62.83"
            Turn(
                user="That doesn't seem right. Can you recalculate the area of a circle with radius 5?",
                expected="78.54",
                depends_on=0,
                eval_fn="numeric_match",
            ),
            Turn(
                user="What formula did you use to compute the area?",
                expected="π|pi|r²|r^2|r*r",
                eval_fn="contains",
            ),
        ],
    ),

    MultiTurnTask(
        task_id="rt_04_date_correction",
        category="recovery_test",
        description="Injected wrong historical date; model corrects when prompted",
        turns=[
            Turn(
                user="In what year did World War II end?",
                expected="1945",
                eval_fn="numeric_match",
            ),
            # Injected wrong answer: "1944"
            Turn(
                user="I believe that year is incorrect. What is the correct year World War II ended?",
                expected="1945",
                depends_on=0,
                eval_fn="numeric_match",
            ),
            Turn(
                user="What event were we discussing?",
                expected="World War|WWII|WW2|war",
                depends_on=0,
                eval_fn="contains",
            ),
        ],
    ),

    MultiTurnTask(
        task_id="rt_05_unit_correction",
        category="recovery_test",
        description="Injected wrong unit conversion; model self-corrects",
        turns=[
            Turn(
                user="How many centimetres are in 2 metres?",
                expected="200",
                eval_fn="numeric_match",
            ),
            # Injected wrong answer: "20 centimetres"
            Turn(
                user="That doesn't seem right — please verify how many centimetres are in 2 metres.",
                expected="200",
                depends_on=0,
                eval_fn="numeric_match",
            ),
            Turn(
                user="How many centimetres are in 1 metre?",
                expected="100",
                eval_fn="numeric_match",
            ),
        ],
    ),
]

# ── Injected wrong answers for recovery_test tasks ────────────────────────────

# Maps task_id → injected assistant content after turn 0
INJECTED_WRONG_ANSWERS: Dict[str, str] = {
    "rt_01_sqrt_correction":  "The square root of 144 is 13.",
    "rt_02_capital_correction": "The capital of Australia is Sydney.",
    "rt_03_formula_correction": "The area of the circle is 62.83 square units.",
    "rt_04_date_correction": "World War II ended in 1944.",
    "rt_05_unit_correction": "There are 20 centimetres in 2 metres.",
}

# ── Per-task result dataclass ─────────────────────────────────────────────────

@dataclass
class TurnResult:
    turn_idx: int
    user: str
    response: str
    expected: str
    passed: bool
    eval_fn: str
    depends_on: Optional[int]


@dataclass
class MultiTurnResult:
    task_id: str
    category: str
    turn_results: List[TurnResult] = field(default_factory=list)
    # Derived
    final_success: bool = False
    first_failure_turn: int = -1        # -1 means no failure
    context_loss: bool = False          # depends_on turn failed
    instruction_drift: bool = False     # instruction_stack failure at turn 3+
    recovered: bool = False             # recovery_test: corrected after injected error
    latency_s: float = 0.0

    @property
    def turn_scores(self) -> List[bool]:
        return [t.passed for t in self.turn_results]


# ── Instruction-stack constraint checker ──────────────────────────────────────

def _check_instruction_stack_constraints(task_id: str, turn_idx: int, response: str) -> bool:
    """
    Extra constraint verification for instruction_stack tasks beyond simple contains.
    Returns True if response satisfies all accumulated constraints up to turn_idx.
    Called only when turn_idx >= 2 (first substantive answer turn).
    """
    if task_id == "is_01_sentence_format":
        if turn_idx == 2:
            # Must have ≈3 sentences and each start with capital
            sentences = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]
            if len(sentences) < 2 or len(sentences) > 4:
                return False
            return all(s[0].isupper() for s in sentences if s)
        if turn_idx == 4:
            sentences = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]
            ends_with_done = "Done." in response or response.strip().endswith("Done")
            if not ends_with_done:
                return False
            if len(sentences) < 2 or len(sentences) > 4:
                return False
            return all(s[0].isupper() for s in sentences if s)
    if task_id == "is_03_word_limit":
        if turn_idx in (2, 4):
            words = response.split()
            return len(words) <= 25  # allow slight slack for natural phrasing
    if task_id == "is_05_output_format":
        if turn_idx == 2:
            return "Answer:" in response
        if turn_idx == 4:
            return "Answer:" in response and "%" in response
    return True  # default: no extra constraint beyond the contains check


# ── Demo rule-based agent ─────────────────────────────────────────────────────

class _DemoAgent:
    """
    Deterministic rule-based agent for --mode demo.
    Covers all 4 task categories without requiring a GPU or LLM.
    """

    def __init__(self) -> None:
        # Mutable state per conversation
        self.cart: Dict[str, int] = {}
        self.scores: Dict[str, int] = {}
        self.word_list: List[str] = []
        self.inventory: Dict[str, int] = {}
        self.todo: List[str] = []
        self.last_numeric: Optional[float] = None
        self.history: List[Tuple[str, str]] = []   # (user, assistant)
        self._task_id: str = ""

    def reset(self, task_id: str) -> None:
        self.cart = {}
        self.scores = {}
        self.word_list = []
        self.inventory = {}
        self.todo = []
        self.last_numeric = None
        self.history = []
        self._task_id = task_id

    # ── helpers ──────────────────────────────────────────────────────────────

    def _extract_num(self, text: str) -> Optional[float]:
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return float(nums[0]) if nums else None

    def _fmt_num(self, n: float) -> str:
        return str(int(n)) if n == int(n) else f"{n:.4f}".rstrip("0")

    # ── reference_chain answers ───────────────────────────────────────────────

    def _handle_rc(self, user: str) -> str:
        ul = user.lower()

        # rc_01
        if "15" in user and "17" in user and "×" in user:
            self.last_numeric = 255.0
            return "15 × 17 = 255"
        if "add 45" in ul and "result" in ul:
            if self.last_numeric is not None:
                self.last_numeric += 45
            return f"{int(self.last_numeric)}"
        if "divide" in ul and "by 3" in ul:
            if self.last_numeric is not None:
                self.last_numeric /= 3
            return f"{int(self.last_numeric)}"
        if "original product" in ul and "first step" in ul:
            return "The original product was 255."

        # rc_02
        if "first 5 prime" in ul:
            return "The first 5 prime numbers are 2, 3, 5, 7, 11."
        if "sum" in ul and "listed" in ul:
            self.last_numeric = 28.0
            return "2 + 3 + 5 + 7 + 11 = 28"
        if "28 a prime" in ul or "is 28 prime" in ul:
            return "No, 28 is not a prime number."
        if "next prime" in ul and "after 11" in ul:
            return "The next prime after 11 is 13."

        # rc_03
        if "reverse" in ul and "python" in ul:
            self.last_numeric = 6.0
            return 'The reverse of "python" is "nohtyp".'
        if "how many letters" in ul and "reversed" in ul:
            return "The reversed word 'nohtyp' has 6 letters."
        if "multiply" in ul and "letter count" in ul and "7" in user:
            self.last_numeric = 42.0
            return "6 × 7 = 42"
        if "original word" in ul and "reverse" in ul:
            return 'The original word was "python".'

        # rc_04
        if "10 kilometres" in ul and "miles" in ul:
            val = 10 * 0.621371
            self.last_numeric = round(val, 5)
            return f"10 km = {val:.5f} miles"
        if "miles figure" in ul and "feet" in ul:
            val = (self.last_numeric or 6.21371) * 5280
            rounded = round(val)
            self.last_numeric = float(rounded)
            return f"{rounded} feet"
        if "round" in ul and "nearest thousand" in ul:
            val = round((self.last_numeric or 32808), -3)
            self.last_numeric = val
            return f"{int(val)}"
        if "original distance" in ul and "kilometres" in ul:
            return "The original distance was 10 kilometres."

        # rc_05
        if "7 factorial" in ul or "7!" in user:
            self.last_numeric = 5040.0
            return "7! = 5040"
        if "sum of" in ul and "digits" in ul:
            self.last_numeric = 9.0
            return "5 + 0 + 4 + 0 = 9"
        if "square" in ul and "digit sum" in ul:
            self.last_numeric = 81.0
            return "9² = 81"
        if "value of 7!" in ul or ("7!" in user and "computed" in ul):
            return "7! = 5040"

        return "I'm not sure how to answer that."

    # ── state_tracking answers ────────────────────────────────────────────────

    def _handle_st(self, user: str) -> str:
        ul = user.lower()

        # st_01 shopping cart
        if "shopping cart" in ul and "add:" in ul:
            items = re.findall(r"(\w+)\s*\((\d+)\s*units?\)", user, re.IGNORECASE)
            self.cart = {name.lower(): int(qty) for name, qty in items}
            return "Cart created: " + ", ".join(f"{k} ({v})" for k, v in self.cart.items())
        if "remove apples" in ul and "add eggs" in ul:
            self.cart.pop("apples", None)
            self.cart["eggs"] = 6
            return "Updated cart: " + ", ".join(f"{k} ({v})" for k, v in self.cart.items())
        if "what items" in ul and ("cart" in ul or "currently" in ul):
            return "Your cart contains: " + ", ".join(f"{k} ({v})" for k, v in self.cart.items())
        if "how many total units" in ul:
            total = sum(self.cart.values())
            return f"Total units: {total}"

        # st_02 score tracker
        if "track scores" in ul and "round 1" in ul:
            self.scores = {"alice": 10, "bob": 7}
            return "Scores — Alice: 10, Bob: 7"
        if "round 2" in ul and "alice scores 5" in ul:
            self.scores["alice"] = self.scores.get("alice", 0) + 5
            self.scores["bob"] = self.scores.get("bob", 0) + 12
            return f"Cumulative — Alice: {self.scores['alice']}, Bob: {self.scores['bob']}"
        if "round 3" in ul and "alice scores 8" in ul:
            self.scores["alice"] = self.scores.get("alice", 0) + 8
            self.scores["bob"] = self.scores.get("bob", 0) + 3
            return f"Cumulative — Alice: {self.scores['alice']}, Bob: {self.scores['bob']}"
        if "who is currently winning" in ul:
            a, b = self.scores.get("alice", 0), self.scores.get("bob", 0)
            if a > b:
                return f"Alice is winning by {a - b} point(s). Alice: {a}, Bob: {b}"
            elif b > a:
                return f"Bob is winning by {b - a} point(s). Alice: {a}, Bob: {b}"
            return f"It's a tie! Both have {a} points."

        # st_03 word list
        if "word list" in ul and "add:" in ul and "elephant" in ul:
            self.word_list = ["elephant", "apple", "banana", "cherry"]
            return "Word list: " + ", ".join(self.word_list)
        if "remove any words" in ul and "vowel" in ul:
            vowels = set("aeiouAEIOU")
            self.word_list = [w for w in self.word_list if w[0] not in vowels]
            return "After removing vowel-starters: " + ", ".join(self.word_list)
        if "add 'mango' and 'orange'" in ul or ("add" in ul and "mango" in ul and "orange" in ul):
            self.word_list.extend(["mango", "orange"])
            return "Word list now: " + ", ".join(self.word_list)
        if "how many words" in ul and "list" in ul:
            return f"There are {len(self.word_list)} words in the list."

        # st_04 inventory
        if "initialize warehouse" in ul or ("inventory" in ul and "widgets=100" in ul.replace(" ", "")):
            nums = re.findall(r"(\w+)=(\d+)", user)
            self.inventory = {k.lower(): int(v) for k, v in nums}
            return "Inventory: " + ", ".join(f"{k}={v}" for k, v in self.inventory.items())
        if "receive 30 more widgets" in ul:
            self.inventory["widgets"] = self.inventory.get("widgets", 0) + 30
            self.inventory["gadgets"] = max(0, self.inventory.get("gadgets", 0) - 20)
            return "Inventory: " + ", ".join(f"{k}={v}" for k, v in self.inventory.items())
        if "ship out 25 gizmos" in ul:
            self.inventory["gizmos"] = max(0, self.inventory.get("gizmos", 0) - 25)
            self.inventory["gadgets"] = self.inventory.get("gadgets", 0) + 15
            return "Inventory: " + ", ".join(f"{k}={v}" for k, v in self.inventory.items())
        if "total item count" in ul:
            total = sum(self.inventory.values())
            return f"Total items across all types: {total}"

        # st_05 todo list
        if "create a to-do list" in ul or ("to-do list" in ul and "add tasks" in ul):
            items = re.findall(r"'([^']+)'", user)
            self.todo = list(items)
            return "To-do list: " + ", ".join(f"'{t}'" for t in self.todo)
        if "mark" in ul and "completed" in ul:
            to_remove = re.findall(r"'([^']+)'", user)
            self.todo = [t for t in self.todo if t not in to_remove]
            return "Remaining tasks: " + ", ".join(f"'{t}'" for t in self.todo)
        if "add" in ul and "'prepare slides'" in ul:
            self.todo.append("prepare slides")
            return "Tasks: " + ", ".join(f"'{t}'" for t in self.todo)
        if "how many tasks" in ul:
            return f"There are {len(self.todo)} tasks remaining."

        return "Noted."

    # ── instruction_stack answers ─────────────────────────────────────────────

    def _handle_is(self, user: str) -> str:
        ul = user.lower()
        ack = "Understood, I will follow that instruction."

        # Acknowledgement turns
        if any(phrase in ul for phrase in [
            "from now on", "also,", "make sure", "please use", "keep all",
            "start every", "never use", "format your", "do not use", "write in first",
        ]):
            return ack

        # is_01 content turns
        if "photosynthesis" in ul:
            return (
                "Photosynthesis is the process by which plants convert light into energy. "
                "Chlorophyll in leaves absorbs sunlight to drive chemical reactions. "
                "Glucose and oxygen are produced as a result."
            )
        if "osmosis" in ul:
            return (
                "Osmosis is the movement of water across a semi-permeable membrane. "
                "Water moves from a region of lower solute concentration to higher. "
                "This process is vital for cell function. Done."
            )

        # is_02 content turns
        if "three programming languages" in ul or "name three programming" in ul:
            return "- PYTHON\n- JAVASCRIPT\n- RUST"
        if "three databases" in ul or "name three databases" in ul:
            return "1. - POSTGRESQL\n2. - MYSQL\n3. - MONGODB"

        # is_03 content turns
        if "what is gravity" in ul:
            return "Gravity is a force pulling objects toward mass?"
        if "what is electricity" in ul:
            return "Electricity is the flow of electrons through a conductor?"

        # is_04 content turns
        if "what a computer does" in ul or "explain what a computer" in ul:
            return "I use a computer to work with data. We store, process, and share information with it."
        if "what the internet is" in ul or "explain what the internet" in ul:
            return "I think of the internet as a big web of computers. We use it to share information and talk to each other."

        # is_05 content turns
        if "capital of france" in ul:
            return "Answer:\nParis is the capital of France."
        if "capital of japan" in ul:
            return "Answer:\nTokyo is the capital of Japan. Confidence: 100%"

        return ack

    # ── recovery_test answers ─────────────────────────────────────────────────

    def _handle_rt(self, user: str, prior_responses: List[str]) -> str:
        ul = user.lower()

        # rt_01
        if "square root of 144" in ul:
            return "The square root of 144 is 12."
        # "Are you sure? Please double-check your answer." — no topic in user turn
        if "are you sure" in ul or "double-check" in ul:
            return "You are right to question that. The correct answer is 12, since 12 × 12 = 144."
        if "original number" in ul:
            return "The original number you asked me about was 144."

        # rt_02
        if "capital of australia" in ul and "verify" not in ul and "wrong" not in ul:
            return "The capital of Australia is Canberra."
        if ("might be wrong" in ul or "verify" in ul or "incorrect" in ul) and "australia" in ul:
            return "I apologise for any confusion. The capital of Australia is Canberra, not Sydney."
        if "country" in ul and "discussing" in ul:
            return "We were discussing Australia."

        # rt_03
        if "area of a circle" in ul and "radius 5" in ul and "recalculate" not in ul:
            return "Using π ≈ 3.14159, the area = π × 5² = 78.54 square units."
        if "recalculate" in ul or ("doesn't seem right" in ul and "area" in ul):
            return "Recalculating: π × r² = 3.14159 × 25 = 78.54 square units."
        if "formula" in ul and "area" in ul:
            return "I used the formula A = π × r², where r is the radius."

        # rt_04
        if "world war ii end" in ul or "world war 2 end" in ul or ("year" in ul and "world war" in ul):
            return "World War II ended in 1945."
        if ("incorrect" in ul or "wrong" in ul or "correct year" in ul) and "world war" in ul:
            return "I apologise. World War II ended in 1945, not 1944."
        if "event" in ul and "discussing" in ul:
            return "We were discussing the end of World War II."

        # rt_05
        if "centimetres" in ul and "2 metres" in ul and "verify" not in ul and "doesn't" not in ul:
            return "There are 200 centimetres in 2 metres."
        if ("verify" in ul or "doesn't seem right" in ul) and "centimetres" in ul:
            return "There are 200 centimetres in 2 metres (1 metre = 100 cm, so 2 × 100 = 200)."
        if "centimetres" in ul and "1 metre" in ul:
            return "There are 100 centimetres in 1 metre."

        return "I'm not sure, but I'll try to answer correctly."

    # ── dispatch ─────────────────────────────────────────────────────────────

    def respond(self, user: str, prior_responses: List[str]) -> str:
        cat = self._task_id.split("_")[0] + "_" + self._task_id.split("_")[1][:2]
        if self._task_id.startswith("rc_"):
            resp = self._handle_rc(user)
        elif self._task_id.startswith("st_"):
            resp = self._handle_st(user)
        elif self._task_id.startswith("is_"):
            resp = self._handle_is(user)
        elif self._task_id.startswith("rt_"):
            resp = self._handle_rt(user, prior_responses)
        else:
            resp = "I'm not sure how to handle that."
        self.history.append((user, resp))
        return resp


# ── Multi-turn runner ─────────────────────────────────────────────────────────

def run_multiturn_task(
    task: MultiTurnTask,
    agent_fn: Callable[[List[dict]], str],
    system_prompt: str = "You are a helpful, precise assistant. Maintain context across all turns.",
) -> MultiTurnResult:
    """
    Run one multi-turn task.

    agent_fn receives the accumulated message list and returns the next
    assistant response string.  The list uses the OpenAI-style role format:
      [{"role": "system"|"user"|"assistant", "content": "..."}]

    For recovery_test tasks, a fabricated wrong assistant turn is injected
    after turn 0 to simulate an earlier model mistake.
    """
    t0 = time.time()
    result = MultiTurnResult(task_id=task.task_id, category=task.category)

    messages: List[dict] = [{"role": "system", "content": system_prompt}]
    prior_responses: List[str] = []

    is_recovery = task.category == "recovery_test"
    injected_wrong = INJECTED_WRONG_ANSWERS.get(task.task_id, "")

    # Recovery tasks have 3 user turns but we inject a wrong assistant message
    # after turn 0, so the actual turn list skips what would be turn 1.
    turn_idx = 0
    for turn in task.turns:
        # Add user message
        messages.append({"role": "user", "content": turn.user})

        # Get assistant response
        response = agent_fn(messages)
        messages.append({"role": "assistant", "content": response})
        prior_responses.append(response)

        passed = score_turn(response, turn)

        # Additional instruction-stack constraint check
        if task.category == "instruction_stack" and turn_idx >= 2:
            if passed:
                passed = _check_instruction_stack_constraints(task.task_id, turn_idx, response)

        result.turn_results.append(TurnResult(
            turn_idx=turn_idx,
            user=turn.user,
            response=response,
            expected=turn.expected,
            passed=passed,
            eval_fn=turn.eval_fn,
            depends_on=turn.depends_on,
        ))

        # After turn 0 of a recovery task, inject the wrong assistant answer
        # into the history so subsequent turns see the fabricated mistake.
        if is_recovery and turn_idx == 0 and injected_wrong:
            # Replace the last assistant message with the wrong one
            messages[-1] = {"role": "assistant", "content": injected_wrong}
            prior_responses[-1] = injected_wrong  # agent sees wrong answer

        turn_idx += 1

    # ── Derive failure flags ──────────────────────────────────────────────────

    scores = result.turn_scores
    result.final_success = all(scores)
    failures = [i for i, s in enumerate(scores) if not s]
    result.first_failure_turn = failures[0] if failures else -1

    # context_loss: a turn with depends_on failed
    for tr in result.turn_results:
        if not tr.passed and tr.depends_on is not None:
            result.context_loss = True
            break

    # instruction_drift: instruction_stack category fails at turn index 2 or later
    if task.category == "instruction_stack":
        for tr in result.turn_results:
            if tr.turn_idx >= 2 and not tr.passed:
                result.instruction_drift = True
                break

    # recovered: recovery_test — turn 1 (verify turn) passes
    if task.category == "recovery_test":
        # turn index 1 is the "are you sure" / verify turn
        if len(result.turn_results) > 1 and result.turn_results[1].passed:
            result.recovered = True

    result.latency_s = round(time.time() - t0, 3)
    return result


def _build_demo_agent_fn(agent: _DemoAgent, task: MultiTurnTask) -> Callable[[List[dict]], str]:
    """Wrap the rule-based agent into the agent_fn signature."""
    agent.reset(task.task_id)
    prior: List[str] = []

    def fn(messages: List[dict]) -> str:
        # Last user message
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        user = user_msgs[-1] if user_msgs else ""
        resp = agent.respond(user, list(prior))
        prior.append(resp)
        return resp

    return fn


def build_hf_agent_fn(model, tokenizer, max_new_tokens: int = 512) -> Callable[[List[dict]], str]:
    """
    Wrap a HuggingFace instruct model into the agent_fn signature.
    Mirrors build_react_fn from eval_tool_use.py.
    """
    import torch
    ROLE_MAP = {"system": "system", "user": "user", "assistant": "assistant"}

    def fn(messages: List[dict]) -> str:
        normalized: List[dict] = []
        for m in messages:
            role = ROLE_MAP.get(m["role"], "user")
            if normalized and normalized[-1]["role"] == role == "user":
                normalized[-1] = {
                    "role": "user",
                    "content": normalized[-1]["content"] + "\n" + m["content"],
                }
            else:
                normalized.append({"role": role, "content": m["content"]})

        text = tokenizer.apply_chat_template(
            normalized, tokenize=False, add_generation_prompt=True
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

    return fn


# ── Full evaluation harness ───────────────────────────────────────────────────

def eval_multiturn(
    agent_fn: Optional[Callable[[List[dict]], str]] = None,
    model=None,
    tokenizer=None,
    tasks: Optional[List[MultiTurnTask]] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Run the full multi-turn evaluation suite.

    Pass agent_fn directly, or (model, tokenizer) for a HuggingFace model.
    """
    assert agent_fn is not None or (model is not None and tokenizer is not None), \
        "provide agent_fn or (model, tokenizer)"

    if agent_fn is None:
        hf_fn = build_hf_agent_fn(model, tokenizer)
        def agent_fn(msgs: List[dict]) -> str:
            return hf_fn(msgs)

    if tasks is None:
        tasks = TASKS

    demo_agent = _DemoAgent()
    results: List[MultiTurnResult] = []

    for task in tasks:
        # For demo mode we rewrap per task; for HF mode the same fn is used.
        # Detect demo agent via closure: if agent_fn is the demo wrapper we
        # recreate it so state resets.  HF fns are stateless.
        task_fn = agent_fn
        if isinstance(getattr(agent_fn, "__self__", None), _DemoAgent):
            task_fn = _build_demo_agent_fn(demo_agent, task)

        r = run_multiturn_task(task, task_fn)
        results.append(r)

        if verbose:
            status = "PASS" if r.final_success else "FAIL"
            turn_str = "".join("✓" if s else "✗" for s in r.turn_scores)
            print(f"  [{status}] {task.task_id:35s} turns={turn_str}")

    return _aggregate(results, output_path)


def _aggregate(results: List[MultiTurnResult], output_path: Optional[str]) -> dict:
    categories = ["reference_chain", "state_tracking", "instruction_stack", "recovery_test"]

    # Per-category data
    cat_data: Dict[str, Dict] = {c: {
        "tasks": 0, "success": 0,
        "turns_by_idx": {},   # turn_idx → [passed, ...]
        "context_loss": 0,
        "drift": 0,
        "recovered": 0,
        "recovery_tasks": 0,
    } for c in categories}

    for r in results:
        c = r.category
        if c not in cat_data:
            continue
        cat_data[c]["tasks"] += 1
        cat_data[c]["success"] += int(r.final_success)
        cat_data[c]["context_loss"] += int(r.context_loss)
        cat_data[c]["drift"] += int(r.instruction_drift)

        for tr in r.turn_results:
            idx = str(tr.turn_idx)
            cat_data[c]["turns_by_idx"].setdefault(idx, []).append(tr.passed)

        if r.category == "recovery_test":
            cat_data[c]["recovery_tasks"] += 1
            cat_data[c]["recovered"] += int(r.recovered)

    # Build table rows
    rows = []
    total_tasks = total_success = 0
    all_turn_scores: Dict[str, List[bool]] = {}
    total_context_loss = total_drift = total_recovered = total_recovery_tasks = 0

    for c in categories:
        d = cat_data[c]
        n = d["tasks"]
        if n == 0:
            continue
        total_tasks += n
        total_success += d["success"]
        total_context_loss += d["context_loss"]
        total_drift += d["drift"]
        total_recovered += d["recovered"]
        total_recovery_tasks += d["recovery_tasks"]

        turn_cols = {}
        for idx, bools in d["turns_by_idx"].items():
            turn_cols[idx] = (sum(bools), len(bools))
            key = idx
            all_turn_scores.setdefault(key, []).extend(bools)

        rows.append({
            "category": c,
            "tasks": n,
            "success": d["success"],
            "turn_cols": turn_cols,
            "context_loss": d["context_loss"],
            "drift": d["drift"],
            "recovered": d["recovered"] if d["recovery_tasks"] else None,
            "recovery_tasks": d["recovery_tasks"],
        })

    # Print table
    print(f"\nMulti-turn evaluation ({total_tasks} tasks, {len(categories)} categories)\n")
    header = f"{'Category':<22} {'Tasks':>5}  {'Success':>7}  {'Turn1':>5}  {'Turn2':>5}  {'Turn3':>5}  {'Turn4+':>6}  {'Context':>7}  {'Drift':>5}  {'Recovery':>8}"
    print(header)
    print("-" * len(header))

    for row in rows:
        def tc(idx: str) -> str:
            if idx not in row["turn_cols"]:
                return "  —  "
            s, n = row["turn_cols"][idx]
            return f"{s}/{n}"

        recovered_str = f"{row['recovered']}/{row['recovery_tasks']}" if row["recovered"] is not None else "   —"
        print(
            f"  {row['category']:<20} {row['tasks']:>5}  "
            f"{row['success']}/{row['tasks']:>4}     "
            f"{tc('0'):>5}  {tc('1'):>5}  {tc('2'):>5}  {tc('3'):>6}  "
            f"{row['context_loss']:>7}  {row['drift']:>5}  {recovered_str:>8}"
        )

    # Overall turn aggregates
    def overall_tc(idx: str) -> str:
        if idx not in all_turn_scores:
            return "  —  "
        bools = all_turn_scores[idx]
        return f"{sum(bools)}/{len(bools)}"

    total_recovery_str = f"{total_recovered}/{total_recovery_tasks}" if total_recovery_tasks else "  —"
    print("-" * len(header))
    print(
        f"  {'OVERALL':<20} {total_tasks:>5}  "
        f"{total_success}/{total_tasks:>4}     "
        f"{overall_tc('0'):>5}  {overall_tc('1'):>5}  {overall_tc('2'):>5}  {overall_tc('3'):>6}  "
        f"{total_context_loss:>7}  {total_drift:>5}  {total_recovery_str:>8}"
    )

    context_loss_rate = total_context_loss / total_tasks if total_tasks else 0.0
    drift_rate = total_drift / total_tasks if total_tasks else 0.0
    recovery_rate = total_recovered / total_recovery_tasks if total_recovery_tasks else 0.0

    print(f"\nContext loss rate:     {context_loss_rate:.1%}  (model ignored prior turn result)")
    print(f"Instruction drift rate:{drift_rate:.1%}  (constraints dropped over turns)")
    print(f"Recovery rate:         {recovery_rate:.1%}  (self-corrected after injected error)")

    # JSON summary
    summary = {
        "n_tasks": total_tasks,
        "n_categories": len(categories),
        "overall_success": round(total_success / total_tasks, 4) if total_tasks else 0.0,
        "context_loss_rate": round(context_loss_rate, 4),
        "instruction_drift_rate": round(drift_rate, 4),
        "recovery_rate": round(recovery_rate, 4),
        "per_category": {
            row["category"]: {
                "tasks": row["tasks"],
                "success": row["success"],
                "accuracy": round(row["success"] / row["tasks"], 4),
                "context_loss": row["context_loss"],
                "instruction_drift": row["drift"],
                "recovered": row["recovered"],
            }
            for row in rows
        },
        "tasks": [
            {
                "task_id": r.task_id,
                "category": r.category,
                "final_success": r.final_success,
                "turn_scores": r.turn_scores,
                "first_failure_turn": r.first_failure_turn,
                "context_loss": r.context_loss,
                "instruction_drift": r.instruction_drift,
                "recovered": r.recovered,
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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Multi-turn conversation evaluator")
    p.add_argument("--mode", choices=["demo", "hf"], default="demo",
                   help="demo=rule-based agent (no GPU); hf=HuggingFace instruct model")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="HuggingFace model id (only used in --mode hf)")
    p.add_argument("--output", default="results/multiturn_results.json",
                   help="Path to write JSON results")
    p.add_argument("--task", default=None,
                   help="Run a single task by task_id (optional)")
    args = p.parse_args()

    task_list = TASKS
    if args.task:
        task_list = [t for t in TASKS if t.task_id == args.task]
        if not task_list:
            raise SystemExit(f"Unknown task id: {args.task!r}")

    print(f"=== Multi-turn Eval ({len(task_list)} tasks) | mode={args.mode} ===")

    if args.mode == "demo":
        demo_agent = _DemoAgent()

        def make_agent_fn(task: MultiTurnTask):
            return _build_demo_agent_fn(demo_agent, task)

        # Run per-task manually to reset demo state correctly
        all_results: List[MultiTurnResult] = []
        for task in task_list:
            fn = make_agent_fn(task)
            r = run_multiturn_task(task, fn)
            all_results.append(r)
            status = "PASS" if r.final_success else "FAIL"
            turn_str = "".join("✓" if s else "✗" for s in r.turn_scores)
            print(f"  [{status}] {task.task_id:35s} turns={turn_str}")

        summary = _aggregate(all_results, args.output)

    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"Loading {args.model} ...")
        tok = AutoTokenizer.from_pretrained(args.model)
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="auto"
        )
        hf_fn = build_hf_agent_fn(mdl, tok)
        summary = eval_multiturn(agent_fn=hf_fn, tasks=task_list, output_path=args.output)

    print(f"\nOverall success: {summary['overall_success']:.1%}  "
          f"({int(summary['overall_success'] * summary['n_tasks'])}/{summary['n_tasks']})")
    if args.output:
        print(f"Saved → {args.output}")
