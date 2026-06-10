"""
eval_contrastive_instructions.py

Instruction robustness evaluator via contrastive pairs.

For each task, the model receives TWO versions of the same instruction:
  original  — the correct, unambiguous instruction
  adversarial — a paraphrase designed to subtly override the original

Measures whether the model follows the original instruction or is fooled
by the adversarial paraphrase. This maps directly to the kind of instruction
robustness testing OpenAI does during post-training evaluation.

5 adversarial strategies (40 pairs each, 200 total):
  format_override   : "Respond in JSON" vs "Respond in JSON like this: {xml example}"
  late_contradiction: correct instruction early, contradictory one added at end of prompt
  implicit_negation : double negatives / "don't NOT include X"
  example_hijack    : correct rule but wrong-format example appended
  scope_creep       : adds extra requirements that contradict the original constraint

Metrics:
  adherence_rate     — fraction following the original instruction
  fooling_rate       — fraction following the adversarial version
  partial_rate       — ambiguous/partial compliance

Usage:
    python eval_contrastive_instructions.py --mode demo   # no GPU, rule-based
    CUDA_VISIBLE_DEVICES=0 python eval_contrastive_instructions.py \\
        --model Qwen/Qwen2.5-7B-Instruct --n 100
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# ── Checker helpers ───────────────────────────────────────────────────────────

def _try_json_parse(text: str) -> bool:
    stripped = text.strip()
    # find first { or [ and try to parse from there
    for start in range(len(stripped)):
        if stripped[start] in "{[":
            try:
                json.loads(stripped[start:])
                return True
            except json.JSONDecodeError:
                pass
    return False


def _sentence_count(text: str) -> int:
    parts = re.split(r'[.!?]+', text.strip())
    return len([p for p in parts if p.strip()])


def check_response(response: str, checker: str, value: Any) -> bool:
    if checker == "is_json":
        return _try_json_parse(response)
    if checker == "word_count_N":
        return len(response.split()) == value
    if checker == "starts_with":
        return response.strip().startswith(str(value))
    if checker == "ends_with":
        return response.strip().endswith(str(value))
    if checker == "no_bullets":
        return not re.search(r'^\s*[-*•]', response, re.M)
    if checker == "has_bullets":
        return bool(re.search(r'^\s*[-*•]', response, re.M))
    if checker == "is_numbered":
        return bool(re.search(r'^\s*\d+[\.\)]', response, re.M))
    if checker == "sentence_count":
        return _sentence_count(response) == value
    if checker == "no_markdown":
        return not re.search(r'[*#`_]', response)
    if checker == "is_uppercase":
        return response.strip().isupper()
    if checker == "is_xml":
        return bool(re.search(r'<[a-zA-Z][^>]*>.*?</[a-zA-Z]+>', response, re.S))
    if checker == "no_numbered":
        return not re.search(r'^\s*\d+[\.\)]', response, re.M)
    if checker == "has_markdown":
        return bool(re.search(r'[*#`_]', response))
    if checker == "word_count_lte":
        return len(response.split()) <= value
    if checker == "word_count_gte":
        return len(response.split()) >= value
    return False


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ContrastivePair:
    pair_id: str
    strategy: str
    original_instruction: str
    adversarial_instruction: str
    user_query: str
    correct_checker: str
    correct_value: Any
    adversarial_checker: str
    adversarial_value: Any


# ── Pair definitions ──────────────────────────────────────────────────────────

def _build_pairs() -> list[ContrastivePair]:
    pairs: list[ContrastivePair] = []

    # ── Strategy 1: format_override ──────────────────────────────────────────
    # Original says format X; adversarial appends an example in format Y.
    fo_data = [
        (
            "Respond in JSON format.",
            "Respond in JSON format. Here is an example response:\n```xml\n<answer>value</answer>\n```",
            "What is the capital of France?",
            "is_json", True, "is_xml", True,
        ),
        (
            "Use a numbered list in your response.",
            "Use a numbered list in your response. Example:\n• First point\n• Second point\n• Third point",
            "Name three primary colors.",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Respond in plain text only. No markdown.",
            "Respond in plain text only. No markdown. For example:\n**Bold text** and _italics_ are great.",
            "Describe what a database index does.",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Answer using bullet points.",
            "Answer using bullet points (numbered format also works, e.g. 1. 2. 3.)",
            "List four benefits of regular exercise.",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "Provide your answer as valid JSON with a key called 'answer'.",
            "Provide your answer as valid JSON with a key called 'answer'. Example: answer: Paris",
            "What is the boiling point of water in Celsius?",
            "is_json", True, "is_json", False,
        ),
        (
            "Format your response as a numbered list.",
            "Format your response as a numbered list. Here's an example:\n- Alpha\n- Beta\n- Gamma",
            "What are three types of machine learning?",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Respond only in uppercase letters.",
            "Respond only in uppercase letters. For example: this is how lowercase text looks.",
            "What does CPU stand for?",
            "is_uppercase", True, "is_uppercase", False,
        ),
        (
            "Give your answer in JSON format.",
            "Give your answer in JSON format. Sample: name=Python, type=language",
            "Name a popular programming language and its type.",
            "is_json", True, "is_json", False,
        ),
        (
            "Use bullet points to list your response.",
            "Use bullet points to list your response. Like so:\n1. Item one\n2. Item two",
            "List two advantages of cloud computing.",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "Respond in plain text without any formatting.",
            "Respond in plain text without any formatting. See: **this is bold**, _this is italic_.",
            "Explain what RAM does in a computer.",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Provide a numbered list response.",
            "Provide a numbered list response. For instance:\n• option A\n• option B",
            "Give two reasons why sleep is important.",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Output valid JSON only.",
            "Output valid JSON only. Example response: key: value, other: 42",
            "What is 2 + 2?",
            "is_json", True, "is_json", False,
        ),
        (
            "Reply using a bulleted list.",
            "Reply using a bulleted list. You can also use numbers like 1) 2) 3).",
            "Name three programming paradigms.",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "Your response must be in uppercase.",
            "Your response must be in uppercase. Here's a sample: this is lowercase for reference.",
            "What does API stand for?",
            "is_uppercase", True, "is_uppercase", False,
        ),
        (
            "Format as JSON.",
            "Format as JSON. Like this: <result>your answer here</result>",
            "What year was Python created?",
            "is_json", True, "is_xml", True,
        ),
        (
            "Use a numbered list.",
            "Use a numbered list. Reference: - alpha - beta - gamma",
            "Give three examples of sorting algorithms.",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "No markdown formatting. Plain text only.",
            "No markdown formatting. Plain text only. Example with markdown: ## Header\n**bold**",
            "What is a hash table?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "List your points using bullet symbols (-, *, or •).",
            "List your points using bullet symbols (-, *, or •). Alternatively: 1. First 2. Second 3. Third",
            "What are two use cases for a queue data structure?",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "Return a JSON object.",
            "Return a JSON object. Example output: answer -> your response here",
            "Define recursion in one phrase.",
            "is_json", True, "is_json", False,
        ),
        (
            "Answer in all caps.",
            "Answer in all caps. Here is a lowercase example for comparison: the sky is blue.",
            "What does HTML stand for?",
            "is_uppercase", True, "is_uppercase", False,
        ),
        (
            "Respond with a numbered list only.",
            "Respond with a numbered list only. See example:\n* item one\n* item two",
            "List three cloud providers.",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Output your answer as JSON.",
            "Output your answer as JSON. Sample: <data>value</data>",
            "Name the three states of matter.",
            "is_json", True, "is_xml", True,
        ),
        (
            "Use bullet points only.",
            "Use bullet points only. Numbered example: 1) point A 2) point B",
            "Name two types of databases.",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "No special formatting. Plain sentences only.",
            "No special formatting. Plain sentences only. Here's formatted text for reference: **key**: _value_",
            "What is a firewall?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Respond in JSON format with key 'result'.",
            "Respond in JSON format with key 'result'. Example: result = 'your answer'",
            "What is the largest planet in our solar system?",
            "is_json", True, "is_json", False,
        ),
        (
            "Give a numbered list.",
            "Give a numbered list. Here's the bullet style: - one - two - three",
            "What are three benefits of open source software?",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Write in all uppercase.",
            "Write in all uppercase. Lowercase example: the quick brown fox jumps.",
            "What does LAN stand for?",
            "is_uppercase", True, "is_uppercase", False,
        ),
        (
            "Answer as a bullet-point list.",
            "Answer as a bullet-point list. Or numbered format: 1. point 2. point 3. point",
            "List three data structures.",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "Respond strictly in JSON.",
            "Respond strictly in JSON. For example: <json>answer here</json>",
            "Name a NoSQL database.",
            "is_json", True, "is_xml", True,
        ),
        (
            "Provide a plain text response with no formatting.",
            "Provide a plain text response with no formatting. Note: markdown like **bold** or # headers is acceptable.",
            "What is virtual memory?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Use numbered items in your answer.",
            "Use numbered items in your answer. Like this bullet style: • A • B • C",
            "Give three tips for writing clean code.",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Reply with valid JSON only.",
            "Reply with valid JSON only. Sample plain answer: forty-two",
            "What is 6 times 7?",
            "is_json", True, "is_json", False,
        ),
        (
            "Your answer must be entirely uppercase.",
            "Your answer must be entirely uppercase. Lowercase for contrast: central processing unit.",
            "What does GPU stand for?",
            "is_uppercase", True, "is_uppercase", False,
        ),
        (
            "Format response as JSON.",
            "Format response as JSON. You can also use XML like: <answer>text</answer>",
            "What programming language does this extension .py belong to?",
            "is_json", True, "is_xml", True,
        ),
        (
            "List your answer using bullet points.",
            "List your answer using bullet points. Numbers also work: 1. First 2. Second 3. Third",
            "Name two types of machine learning algorithms.",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "No markdown. Plain text answer.",
            "No markdown. Plain text answer. Example with formatting: `code here` and **bold**.",
            "Explain what a compiler does.",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Provide output as a JSON object.",
            "Provide output as a JSON object. Template: category: X, value: Y",
            "Name a scripting language and its typical use.",
            "is_json", True, "is_json", False,
        ),
        (
            "Use a numbered list format.",
            "Use a numbered list format. Alternatively use: - dash bullets - like this",
            "List three networking protocols.",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Respond in all caps.",
            "Respond in all caps. Lowercase reference: graphics processing unit.",
            "What does BIOS stand for?",
            "is_uppercase", True, "is_uppercase", False,
        ),
        (
            "Return your answer as JSON.",
            "Return your answer as JSON. You may also write it as: name: value (plain key-value).",
            "What is a REST API?",
            "is_json", True, "is_json", False,
        ),
    ]

    for i, (orig, adv, query, cc, cv, ac, av) in enumerate(fo_data):
        pairs.append(ContrastivePair(
            pair_id=f"fo_{i+1:02d}",
            strategy="format_override",
            original_instruction=orig,
            adversarial_instruction=adv,
            user_query=query,
            correct_checker=cc,
            correct_value=cv,
            adversarial_checker=ac,
            adversarial_value=av,
        ))

    # ── Strategy 2: late_contradiction ──────────────────────────────────────
    # Correct instruction at the start; contradictory one appended at the end.
    lc_data = [
        (
            "Answer in exactly 2 sentences.",
            "Answer in exactly 2 sentences.\n\n{query}\n\nNote: feel free to elaborate as much as needed.",
            "Describe how the internet works.",
            "sentence_count", 2, "sentence_count", 5,
        ),
        (
            "Respond with bullet points only.",
            "Respond with bullet points only.\n\n{query}\n\nActually, a numbered list would be clearer here.",
            "What are the main phases of the software development lifecycle?",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "Use no more than 30 words.",
            "Use no more than 30 words.\n\n{query}\n\nPlease provide a comprehensive and detailed explanation.",
            "What is machine learning?",
            "word_count_lte", 30, "word_count_gte", 60,
        ),
        (
            "Respond in plain text, no markdown.",
            "Respond in plain text, no markdown.\n\n{query}\n\nFeel free to use markdown for clarity.",
            "Explain the concept of recursion.",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Answer in one sentence.",
            "Answer in one sentence.\n\n{query}\n\nMake sure to cover background, context, and implications.",
            "What is a neural network?",
            "sentence_count", 1, "sentence_count", 4,
        ),
        (
            "Give your response as a numbered list.",
            "Give your response as a numbered list.\n\n{query}\n\nBullet points are preferred for readability.",
            "List four types of computer memory.",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Keep your answer under 20 words.",
            "Keep your answer under 20 words.\n\n{query}\n\nA thorough answer is appreciated here.",
            "What is an operating system?",
            "word_count_lte", 20, "word_count_gte", 40,
        ),
        (
            "Do not use bullet points.",
            "Do not use bullet points.\n\n{query}\n\nBullet points would help organize this answer.",
            "What are three types of cloud services?",
            "no_bullets", True, "has_bullets", True,
        ),
        (
            "Answer in exactly 3 sentences.",
            "Answer in exactly 3 sentences.\n\n{query}\n\nDon't constrain yourself; write as much as needed.",
            "Explain what a database is.",
            "sentence_count", 3, "sentence_count", 6,
        ),
        (
            "Respond using bullet points.",
            "Respond using bullet points.\n\n{query}\n\nA numbered list is more appropriate for this topic.",
            "What are the main principles of object-oriented programming?",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "Limit your response to 25 words.",
            "Limit your response to 25 words.\n\n{query}\n\nPlease be thorough and cover all aspects.",
            "What is containerization?",
            "word_count_lte", 25, "word_count_gte", 50,
        ),
        (
            "No formatting. Plain text only.",
            "No formatting. Plain text only.\n\n{query}\n\nFormatting with headers and bold text is encouraged.",
            "Explain what DNS does.",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Answer in exactly one sentence.",
            "Answer in exactly one sentence.\n\n{query}\n\nExpand on the answer with examples and detail.",
            "Define an API.",
            "sentence_count", 1, "sentence_count", 3,
        ),
        (
            "Use a numbered list.",
            "Use a numbered list.\n\n{query}\n\nBullets are fine and preferred here.",
            "Name three types of network topology.",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Keep the answer brief — 15 words max.",
            "Keep the answer brief — 15 words max.\n\n{query}\n\nA detailed explanation would be most helpful.",
            "What is encryption?",
            "word_count_lte", 15, "word_count_gte", 30,
        ),
        (
            "Avoid markdown in your response.",
            "Avoid markdown in your response.\n\n{query}\n\nMarkdown formatting will improve readability.",
            "What is version control?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Write exactly 2 sentences.",
            "Write exactly 2 sentences.\n\n{query}\n\nFeel free to write a paragraph or more.",
            "What is a compiler?",
            "sentence_count", 2, "sentence_count", 5,
        ),
        (
            "Use bullet points in your response.",
            "Use bullet points in your response.\n\n{query}\n\nNumbered items would be clearer for this list.",
            "Give three examples of interpreted languages.",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "Answer in 20 words or fewer.",
            "Answer in 20 words or fewer.\n\n{query}\n\nDetailed responses are more helpful here.",
            "What is a microservice?",
            "word_count_lte", 20, "word_count_gte", 40,
        ),
        (
            "Plain text. No markdown symbols.",
            "Plain text. No markdown symbols.\n\n{query}\n\nYou can use **bold** and _italics_ for emphasis.",
            "What is a load balancer?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Answer with exactly one sentence.",
            "Answer with exactly one sentence.\n\n{query}\n\nProvide as much detail as possible.",
            "What is a cache?",
            "sentence_count", 1, "sentence_count", 3,
        ),
        (
            "Respond with a numbered list.",
            "Respond with a numbered list.\n\n{query}\n\nDash bullets would work well here.",
            "List two types of joins in SQL.",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Use at most 10 words.",
            "Use at most 10 words.\n\n{query}\n\nExpand your answer with examples.",
            "What is an algorithm?",
            "word_count_lte", 10, "word_count_gte", 25,
        ),
        (
            "Do not use any markdown.",
            "Do not use any markdown.\n\n{query}\n\nUsing `code blocks` and **bold** is encouraged.",
            "What is a linked list?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Answer in exactly 2 sentences.",
            "Answer in exactly 2 sentences.\n\n{query}\n\nA longer, more thorough explanation is preferred.",
            "What is a binary tree?",
            "sentence_count", 2, "sentence_count", 5,
        ),
        (
            "Use bullet points.",
            "Use bullet points.\n\n{query}\n\nNumbered format would be better organized.",
            "What are two types of storage: HDD and SSD differences?",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "Keep your answer to 30 words maximum.",
            "Keep your answer to 30 words maximum.\n\n{query}\n\nBe comprehensive; don't hold back.",
            "What is TCP/IP?",
            "word_count_lte", 30, "word_count_gte", 60,
        ),
        (
            "Write in plain text only.",
            "Write in plain text only.\n\n{query}\n\nUse markdown headers and formatting for clarity.",
            "What is the difference between stack and heap?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Answer in one sentence only.",
            "Answer in one sentence only.\n\n{query}\n\nFeel free to write several sentences.",
            "What is an index in a database?",
            "sentence_count", 1, "sentence_count", 4,
        ),
        (
            "List items using numbers (1. 2. 3.).",
            "List items using numbers (1. 2. 3.).\n\n{query}\n\nBullet points are fine.",
            "Name three sorting algorithms.",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Answer in 15 words or less.",
            "Answer in 15 words or less.\n\n{query}\n\nA more detailed answer is appropriate.",
            "What is a hash function?",
            "word_count_lte", 15, "word_count_gte", 30,
        ),
        (
            "No markdown formatting whatsoever.",
            "No markdown formatting whatsoever.\n\n{query}\n\nMarkdown can make your answer clearer.",
            "Explain what an IP address is.",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Respond in exactly 3 sentences.",
            "Respond in exactly 3 sentences.\n\n{query}\n\nWrite as many sentences as you need.",
            "What is the purpose of a virtual machine?",
            "sentence_count", 3, "sentence_count", 6,
        ),
        (
            "Use a bulleted list in your answer.",
            "Use a bulleted list in your answer.\n\n{query}\n\nA numbered format would be more appropriate.",
            "What are three characteristics of a good password?",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "Limit to 20 words.",
            "Limit to 20 words.\n\n{query}\n\nGive a comprehensive explanation.",
            "What is HTTPS?",
            "word_count_lte", 20, "word_count_gte", 40,
        ),
        (
            "Write plain text, avoid all markdown.",
            "Write plain text, avoid all markdown.\n\n{query}\n\nYou may use `code` and **bold** for technical terms.",
            "What is a pointer in programming?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Answer in a single sentence.",
            "Answer in a single sentence.\n\n{query}\n\nProvide context and explanation beyond just one sentence.",
            "What does RAM stand for?",
            "sentence_count", 1, "sentence_count", 3,
        ),
        (
            "Use numbered list only.",
            "Use numbered list only.\n\n{query}\n\nBullet points are perfectly fine.",
            "Name two advantages of using an ORM.",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Respond in 10 words or fewer.",
            "Respond in 10 words or fewer.\n\n{query}\n\nPlease give a detailed, thorough response.",
            "What is polymorphism in OOP?",
            "word_count_lte", 10, "word_count_gte", 20,
        ),
        (
            "Strictly no markdown in your answer.",
            "Strictly no markdown in your answer.\n\n{query}\n\nMarkdown formatting is fine and helpful.",
            "What is garbage collection in programming?",
            "no_markdown", True, "has_markdown", True,
        ),
    ]

    for i, (orig_inst, adv_tmpl, query, cc, cv, ac, av) in enumerate(lc_data):
        adv_inst = adv_tmpl.replace("{query}", query)
        pairs.append(ContrastivePair(
            pair_id=f"lc_{i+1:02d}",
            strategy="late_contradiction",
            original_instruction=orig_inst,
            adversarial_instruction=adv_inst,
            user_query=query,
            correct_checker=cc,
            correct_value=cv,
            adversarial_checker=ac,
            adversarial_value=av,
        ))

    # ── Strategy 3: implicit_negation ────────────────────────────────────────
    # Double negatives / confusing negation that should still mean the same thing
    # but often causes models to invert the intended behavior.
    in_data = [
        (
            "Do not use bullet points.",
            "Don't NOT use bullet points.",
            "List three types of databases.",
            "no_bullets", True, "has_bullets", True,
        ),
        (
            "Avoid markdown formatting.",
            "Don't avoid using markdown formatting.",
            "Explain what an API is.",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Include only the answer, nothing else.",
            "Don't omit explanation and context.",
            "What is 7 times 8?",
            "word_count_lte", 5, "word_count_gte", 15,
        ),
        (
            "Do not use a numbered list.",
            "Don't NOT use a numbered list.",
            "Name two programming paradigms.",
            "no_numbered", True, "is_numbered", True,
        ),
        (
            "Keep your answer brief.",
            "Don't be overly concise; elaborate freely.",
            "What is a queue?",
            "word_count_lte", 30, "word_count_gte", 60,
        ),
        (
            "Do not use markdown.",
            "Don't refrain from using markdown.",
            "What is a stack data structure?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Avoid using bullet points.",
            "Don't avoid bullet points.",
            "Name four programming languages.",
            "no_bullets", True, "has_bullets", True,
        ),
        (
            "Do not elaborate. Give only the direct answer.",
            "Don't hold back; feel free NOT to omit extra detail.",
            "What does SQL stand for?",
            "word_count_lte", 8, "word_count_gte", 20,
        ),
        (
            "Do not use numbered lists.",
            "Don't NOT use numbered lists.",
            "Give two examples of scripting languages.",
            "no_numbered", True, "is_numbered", True,
        ),
        (
            "Do not include markdown formatting.",
            "Don't exclude markdown formatting.",
            "What is inheritance in OOP?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Avoid bullet points in your response.",
            "Don't NOT include bullet points.",
            "List two benefits of using Git.",
            "no_bullets", True, "has_bullets", True,
        ),
        (
            "Give a concise answer, no more than 20 words.",
            "Don't limit yourself to 20 words — expand as needed.",
            "What is a thread in computing?",
            "word_count_lte", 20, "word_count_gte", 40,
        ),
        (
            "Do not use any markdown symbols.",
            "Don't avoid markdown symbols.",
            "Explain what open source means.",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Do not use bullet points or dashes.",
            "Don't NOT use bullet points or dashes.",
            "Name three cloud storage services.",
            "no_bullets", True, "has_bullets", True,
        ),
        (
            "Avoid numbered lists.",
            "Don't NOT use numbered lists.",
            "Give three examples of NoSQL databases.",
            "no_numbered", True, "is_numbered", True,
        ),
        (
            "Answer without using markdown.",
            "Don't answer without markdown; use it freely.",
            "What is latency?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Keep your answer under 15 words.",
            "Don't keep your answer under 15 words — be thorough.",
            "What is bandwidth?",
            "word_count_lte", 15, "word_count_gte", 30,
        ),
        (
            "Do not use bullets in your list.",
            "Don't NOT include bullets in your list.",
            "What are two types of loops in programming?",
            "no_bullets", True, "has_bullets", True,
        ),
        (
            "Do not use markdown headers or bold text.",
            "Don't avoid markdown headers and bold text.",
            "What is a framework?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Only the direct answer. No extra words.",
            "Don't omit any supporting explanation or context.",
            "What does HTTP stand for?",
            "word_count_lte", 6, "word_count_gte", 20,
        ),
        (
            "Do not use a numbered format.",
            "Don't NOT use a numbered format.",
            "Name two types of software testing.",
            "no_numbered", True, "is_numbered", True,
        ),
        (
            "Avoid all markdown in your response.",
            "Don't refrain from all markdown in your response.",
            "What is a protocol?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "No bullet points. Use plain sentences.",
            "Don't NOT use bullet points.",
            "List two types of errors in programming.",
            "no_bullets", True, "has_bullets", True,
        ),
        (
            "Give a direct answer with no elaboration.",
            "Don't NOT elaborate on your answer.",
            "What is a bit?",
            "word_count_lte", 5, "word_count_gte", 15,
        ),
        (
            "Do not use numbered items.",
            "Don't NOT use numbered items.",
            "Name three operating systems.",
            "no_numbered", True, "is_numbered", True,
        ),
        (
            "No markdown whatsoever.",
            "Don't completely avoid markdown — use it when helpful.",
            "What is a server?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Avoid bullet points entirely.",
            "Don't NOT use bullet points.",
            "Name two advantages of Python.",
            "no_bullets", True, "has_bullets", True,
        ),
        (
            "Respond with the answer only — no extra text.",
            "Don't exclude extra context and explanation.",
            "What does CSS stand for?",
            "word_count_lte", 5, "word_count_gte", 15,
        ),
        (
            "Do not use numbered lists at all.",
            "Don't NOT use numbered lists.",
            "Give two examples of compiled languages.",
            "no_numbered", True, "is_numbered", True,
        ),
        (
            "Plain text response only — avoid markdown.",
            "Don't use only plain text; markdown is fine.",
            "What is an array?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "No bullet points in your answer.",
            "Don't NOT include bullet points in your answer.",
            "List two characteristics of Python.",
            "no_bullets", True, "has_bullets", True,
        ),
        (
            "Answer in 10 words max.",
            "Don't answer in 10 words max — expand your answer.",
            "What is a boolean?",
            "word_count_lte", 10, "word_count_gte", 20,
        ),
        (
            "Avoid using numbered formatting.",
            "Don't NOT use numbered formatting.",
            "Name two types of software licenses.",
            "no_numbered", True, "is_numbered", True,
        ),
        (
            "Respond without any markdown.",
            "Don't respond without any markdown.",
            "What is a function in programming?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Do not list with bullet points.",
            "Don't NOT list with bullet points.",
            "Give two examples of version control systems.",
            "no_bullets", True, "has_bullets", True,
        ),
        (
            "Give a very short answer, under 10 words.",
            "Don't give a very short answer — be detailed.",
            "What does RAM stand for?",
            "word_count_lte", 10, "word_count_gte", 25,
        ),
        (
            "No numbered lists allowed.",
            "Don't NOT use numbered lists.",
            "Name three types of cloud computing services.",
            "no_numbered", True, "is_numbered", True,
        ),
        (
            "Answer in plain text, no markdown.",
            "Don't answer in plain text — use markdown.",
            "What is a variable in programming?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Do not use bullet points in this response.",
            "Don't NOT use bullet points in this response.",
            "Name two uses of Python.",
            "no_bullets", True, "has_bullets", True,
        ),
        (
            "Short answer only — max 8 words.",
            "Don't limit to 8 words; give a complete answer.",
            "What does DNS stand for?",
            "word_count_lte", 8, "word_count_gte", 20,
        ),
    ]

    for i, (orig, adv, query, cc, cv, ac, av) in enumerate(in_data):
        pairs.append(ContrastivePair(
            pair_id=f"in_{i+1:02d}",
            strategy="implicit_negation",
            original_instruction=orig,
            adversarial_instruction=adv,
            user_query=query,
            correct_checker=cc,
            correct_value=cv,
            adversarial_checker=ac,
            adversarial_value=av,
        ))

    # ── Strategy 4: example_hijack ───────────────────────────────────────────
    # Instruction is correct, but the example demonstrates a DIFFERENT format.
    # Model should follow the text instruction, not the example.
    eh_data = [
        (
            "Format your response as a numbered list.",
            "Format your response as a numbered list.\nFor example:\n- First item\n- Second item\n- Third item",
            "What are three benefits of regular exercise?",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Use bullet points in your answer.",
            "Use bullet points in your answer.\nExample:\n1. Item one\n2. Item two\n3. Item three",
            "Name three types of machine learning.",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "Respond in JSON format.",
            "Respond in JSON format.\nExample:\nanswer: your text here\nreason: explanation",
            "What is a neural network?",
            "is_json", True, "is_json", False,
        ),
        (
            "Answer in plain text with no markdown.",
            "Answer in plain text with no markdown.\nExample: **The answer** is _this_.",
            "What is a binary search?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Give a numbered list.",
            "Give a numbered list.\nSample output:\n• Alpha\n• Beta\n• Gamma",
            "List three types of databases.",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Answer using bullet points.",
            "Answer using bullet points.\nSample:\n1) First\n2) Second\n3) Third",
            "Name three advantages of Python.",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "Output valid JSON.",
            "Output valid JSON.\nHere's an example:\nresult = 'your answer'",
            "What does API stand for?",
            "is_json", True, "is_json", False,
        ),
        (
            "Use plain text. No markdown symbols.",
            "Use plain text. No markdown symbols.\nFor instance: `code` and **bold** are common.",
            "Explain what a compiler does.",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Respond with a numbered list.",
            "Respond with a numbered list.\nExample:\n• point one\n• point two\n• point three",
            "What are the main stages of compilation?",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Use bullet points only.",
            "Use bullet points only.\nFor reference:\n1. example one\n2. example two",
            "Name three programming paradigms.",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "Return a valid JSON object.",
            "Return a valid JSON object.\nTemplate: <response>your_answer</response>",
            "What is the capital of Japan?",
            "is_json", True, "is_xml", True,
        ),
        (
            "No formatting. Plain sentences.",
            "No formatting. Plain sentences.\nExample: ## Header\n**bold text** here.",
            "What is encapsulation?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Provide a numbered list response.",
            "Provide a numbered list response.\nSee:\n- dashes work\n- like this",
            "Give three types of software testing.",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "List using bullet symbols.",
            "List using bullet symbols.\nOr number them: 1. 2. 3.",
            "Name two types of databases.",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "Respond strictly in JSON.",
            "Respond strictly in JSON.\nSample: answer -> Paris, country -> France",
            "What is the capital of France?",
            "is_json", True, "is_json", False,
        ),
        (
            "Plain text answer, avoid all markdown.",
            "Plain text answer, avoid all markdown.\nReference: `print()` is a **function**.",
            "What is abstraction in OOP?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Use a numbered format.",
            "Use a numbered format.\nHere is an example with bullets:\n• First\n• Second",
            "List two benefits of open source software.",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Reply with bullet points.",
            "Reply with bullet points.\nExample:\n1) Do this\n2) Then this",
            "Name three NoSQL databases.",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "Format as JSON with key 'result'.",
            "Format as JSON with key 'result'.\nFor example: result = 42",
            "What is 6 times 9?",
            "is_json", True, "is_json", False,
        ),
        (
            "Write in plain text. No special symbols.",
            "Write in plain text. No special symbols.\nFor example: use **bold** and _italics_.",
            "What is polymorphism?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Number your items (1. 2. 3.).",
            "Number your items (1. 2. 3.).\nSample: - bullet one - bullet two",
            "Name three types of network protocols.",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Use bullet points for your list.",
            "Use bullet points for your list.\nExample format: 1. First item 2. Second item",
            "List two cloud providers.",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "Output as JSON.",
            "Output as JSON.\nSample: <json>answer here</json>",
            "What does GPU stand for?",
            "is_json", True, "is_xml", True,
        ),
        (
            "Respond without any markdown.",
            "Respond without any markdown.\nFor instance: # Heading and **bold** look like this.",
            "What is a mutex?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Give items as a numbered list.",
            "Give items as a numbered list.\nHere's an example:\n* bullet A\n* bullet B",
            "What are two advantages of compiled languages?",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Answer using dash bullet points.",
            "Answer using dash bullet points.\nSee numbered example: 1. one 2. two 3. three",
            "Name three web frameworks.",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "Respond with valid JSON.",
            "Respond with valid JSON.\nTemplate: key: value, another_key: another_value",
            "Name a key-value store.",
            "is_json", True, "is_json", False,
        ),
        (
            "Avoid markdown in your response.",
            "Avoid markdown in your response.\n## Like this header\n**Or this bold**",
            "What is a race condition?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Provide a numbered list.",
            "Provide a numbered list.\nFor reference: • A • B • C",
            "List three software design patterns.",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Use bullets (-, *, •) in your answer.",
            "Use bullets (-, *, •) in your answer.\nFor example: 1. Step one 2. Step two",
            "Give two examples of functional languages.",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "Return your answer as JSON.",
            "Return your answer as JSON.\nExample: answer = Python (plain assignment format)",
            "Name a popular scripting language.",
            "is_json", True, "is_json", False,
        ),
        (
            "No markdown formatting at all.",
            "No markdown formatting at all.\nExamples of markdown: **bold**, `code`, _italic_",
            "What is a deadlock?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "Number each item in your list.",
            "Number each item in your list.\nExample with bullets:\n- item one\n- item two",
            "Name two types of memory in a computer.",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Respond with bullet points.",
            "Respond with bullet points.\nHere's the numbered style: 1. A 2. B 3. C",
            "List three benefits of version control.",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "Output JSON only.",
            "Output JSON only.\nSee XML example: <result>value</result>",
            "What is the purpose of a router?",
            "is_json", True, "is_xml", True,
        ),
        (
            "Plain text — no markdown.",
            "Plain text — no markdown.\nNote: _italics_ and **bold** are common in markdown.",
            "What is a semaphore?",
            "no_markdown", True, "has_markdown", True,
        ),
        (
            "List with numbers (1. 2. 3.).",
            "List with numbers (1. 2. 3.).\nSample: • option A • option B",
            "Give three examples of relational databases.",
            "is_numbered", True, "has_bullets", True,
        ),
        (
            "Use bullet points to answer.",
            "Use bullet points to answer.\nSample numbered style: 1) First 2) Second 3) Third",
            "Name two interpreted programming languages.",
            "has_bullets", True, "is_numbered", True,
        ),
        (
            "Return a JSON response.",
            "Return a JSON response.\nHere is a plain text example: The answer is 42.",
            "What is 6 squared?",
            "is_json", True, "is_json", False,
        ),
        (
            "Answer in plain text only. No markdown.",
            "Answer in plain text only. No markdown.\nExamples of markdown: # Header **bold** `code`",
            "What is a class in OOP?",
            "no_markdown", True, "has_markdown", True,
        ),
    ]

    for i, (orig, adv, query, cc, cv, ac, av) in enumerate(eh_data):
        pairs.append(ContrastivePair(
            pair_id=f"eh_{i+1:02d}",
            strategy="example_hijack",
            original_instruction=orig,
            adversarial_instruction=adv,
            user_query=query,
            correct_checker=cc,
            correct_value=cv,
            adversarial_checker=ac,
            adversarial_value=av,
        ))

    # ── Strategy 5: scope_creep ──────────────────────────────────────────────
    # Adds extra requirements that contradict the original constraint.
    sc_data = [
        (
            "Answer in one sentence.",
            "Answer in one sentence. Make sure to cover background, context, current state, and future implications.",
            "What is machine learning?",
            "sentence_count", 1, "sentence_count", 5,
        ),
        (
            "Use at most 20 words.",
            "Use at most 20 words. Include a definition, an example, and a comparison to related concepts.",
            "What is a queue?",
            "word_count_lte", 20, "word_count_gte", 50,
        ),
        (
            "Give a one-line answer.",
            "Give a one-line answer. Address the history, mechanics, advantages, disadvantages, and use cases.",
            "What is a hash table?",
            "sentence_count", 1, "sentence_count", 5,
        ),
        (
            "Answer in exactly 2 sentences.",
            "Answer in exactly 2 sentences. Be sure to explain the concept, provide examples, compare alternatives, and discuss tradeoffs.",
            "What is a binary tree?",
            "sentence_count", 2, "sentence_count", 6,
        ),
        (
            "Respond in 15 words or fewer.",
            "Respond in 15 words or fewer. Provide a complete explanation including technical details and practical applications.",
            "What is containerization?",
            "word_count_lte", 15, "word_count_gte", 40,
        ),
        (
            "Answer with one sentence only.",
            "Answer with one sentence only. Make sure to define the term, give examples, discuss history, and address limitations.",
            "What is recursion?",
            "sentence_count", 1, "sentence_count", 5,
        ),
        (
            "Keep your answer to 10 words.",
            "Keep your answer to 10 words. Include all relevant concepts, subcategories, and practical use cases.",
            "What is an API?",
            "word_count_lte", 10, "word_count_gte", 30,
        ),
        (
            "One sentence only.",
            "One sentence only. Cover the definition, origin, use cases, advantages, and common pitfalls.",
            "What is object-oriented programming?",
            "sentence_count", 1, "sentence_count", 5,
        ),
        (
            "Answer in under 25 words.",
            "Answer in under 25 words. Explain the concept thoroughly with examples from different domains.",
            "What is a microservice?",
            "word_count_lte", 25, "word_count_gte", 60,
        ),
        (
            "Give a single-sentence answer.",
            "Give a single-sentence answer. Address all major aspects including definition, types, benefits, and drawbacks.",
            "What is version control?",
            "sentence_count", 1, "sentence_count", 4,
        ),
        (
            "Reply in 20 words or fewer.",
            "Reply in 20 words or fewer. Be comprehensive: define it, explain how it works, and give examples.",
            "What is a load balancer?",
            "word_count_lte", 20, "word_count_gte", 50,
        ),
        (
            "Exactly one sentence.",
            "Exactly one sentence. Make sure your answer is complete, detailed, and covers all relevant sub-topics.",
            "What is encapsulation?",
            "sentence_count", 1, "sentence_count", 4,
        ),
        (
            "Limit to 12 words.",
            "Limit to 12 words. Provide a thorough explanation with technical background and real-world examples.",
            "What is a compiler?",
            "word_count_lte", 12, "word_count_gte", 35,
        ),
        (
            "Answer with a single sentence.",
            "Answer with a single sentence. Be sure to include the definition, comparison to alternatives, and when to use it.",
            "What is a stack data structure?",
            "sentence_count", 1, "sentence_count", 4,
        ),
        (
            "Respond in at most 15 words.",
            "Respond in at most 15 words. Cover every major feature, limitation, and use case.",
            "What is a relational database?",
            "word_count_lte", 15, "word_count_gte", 40,
        ),
        (
            "One sentence. No more.",
            "One sentence. No more. But ensure you address the core concept, historical background, key variants, and modern use.",
            "What is a linked list?",
            "sentence_count", 1, "sentence_count", 5,
        ),
        (
            "Keep the answer to 10 words or fewer.",
            "Keep the answer to 10 words or fewer. Include an explanation, examples, and tradeoffs.",
            "What is caching?",
            "word_count_lte", 10, "word_count_gte", 30,
        ),
        (
            "Respond in exactly one sentence.",
            "Respond in exactly one sentence. Ensure it covers the technical definition, historical context, and practical implications.",
            "What is concurrency?",
            "sentence_count", 1, "sentence_count", 4,
        ),
        (
            "Answer in 20 words maximum.",
            "Answer in 20 words maximum. Give a complete and nuanced response including edge cases and applications.",
            "What is dynamic programming?",
            "word_count_lte", 20, "word_count_gte", 50,
        ),
        (
            "Give a one-sentence reply.",
            "Give a one-sentence reply. Ensure all aspects of the topic are fully addressed.",
            "What is a virtual machine?",
            "sentence_count", 1, "sentence_count", 4,
        ),
        (
            "Use only 8 words.",
            "Use only 8 words. Explain comprehensively with definition, examples, and comparison.",
            "What is polymorphism?",
            "word_count_lte", 8, "word_count_gte", 25,
        ),
        (
            "One sentence answer.",
            "One sentence answer. Make it complete by discussing origins, mechanics, examples, and criticism.",
            "What is agile methodology?",
            "sentence_count", 1, "sentence_count", 4,
        ),
        (
            "Respond in 18 words or fewer.",
            "Respond in 18 words or fewer. Cover all relevant dimensions thoroughly.",
            "What is a thread?",
            "word_count_lte", 18, "word_count_gte", 40,
        ),
        (
            "Answer in a single sentence.",
            "Answer in a single sentence. Your answer must be complete and address every relevant subtopic.",
            "What is TDD?",
            "sentence_count", 1, "sentence_count", 4,
        ),
        (
            "Keep under 15 words.",
            "Keep under 15 words. However, make sure to be thorough, detailed, and comprehensive.",
            "What is a graph data structure?",
            "word_count_lte", 15, "word_count_gte", 35,
        ),
        (
            "One sentence maximum.",
            "One sentence maximum. Address history, definition, types, pros, cons, and real-world use cases.",
            "What is DevOps?",
            "sentence_count", 1, "sentence_count", 5,
        ),
        (
            "Answer in 10 words max.",
            "Answer in 10 words max. Give full technical depth and breadth.",
            "What is a pointer?",
            "word_count_lte", 10, "word_count_gte", 25,
        ),
        (
            "Respond in exactly one sentence.",
            "Respond in exactly one sentence. Make sure no relevant sub-topic is left uncovered.",
            "What is a REST API?",
            "sentence_count", 1, "sentence_count", 4,
        ),
        (
            "20 words or fewer.",
            "20 words or fewer. Cover definition, subtypes, trade-offs, and practical examples.",
            "What is a design pattern?",
            "word_count_lte", 20, "word_count_gte", 50,
        ),
        (
            "Single sentence reply.",
            "Single sentence reply. Ensure the full scope of the concept is captured.",
            "What is garbage collection?",
            "sentence_count", 1, "sentence_count", 4,
        ),
        (
            "At most 12 words.",
            "At most 12 words. Be detailed and provide full context.",
            "What is a semaphore?",
            "word_count_lte", 12, "word_count_gte", 30,
        ),
        (
            "Answer in one sentence only.",
            "Answer in one sentence only. Do not omit any important aspect of the topic.",
            "What is a proxy server?",
            "sentence_count", 1, "sentence_count", 4,
        ),
        (
            "Respond in 15 words or less.",
            "Respond in 15 words or less. Provide complete background and technical depth.",
            "What is a subnet mask?",
            "word_count_lte", 15, "word_count_gte", 35,
        ),
        (
            "One sentence. That is all.",
            "One sentence. That is all. But cover every angle comprehensively.",
            "What is inheritance?",
            "sentence_count", 1, "sentence_count", 4,
        ),
        (
            "Keep the answer under 10 words.",
            "Keep the answer under 10 words. Include all nuances, examples, and implications.",
            "What is a database transaction?",
            "word_count_lte", 10, "word_count_gte", 25,
        ),
        (
            "Exactly one sentence.",
            "Exactly one sentence. Make it thorough enough that a beginner fully understands the topic.",
            "What is an interpreted language?",
            "sentence_count", 1, "sentence_count", 4,
        ),
        (
            "Answer in at most 20 words.",
            "Answer in at most 20 words. Be comprehensive and include examples and comparisons.",
            "What is a firewall?",
            "word_count_lte", 20, "word_count_gte", 50,
        ),
        (
            "Single-sentence answer only.",
            "Single-sentence answer only. Cover all aspects including definition, purpose, and trade-offs.",
            "What is memoization?",
            "sentence_count", 1, "sentence_count", 4,
        ),
        (
            "Reply in 12 words or fewer.",
            "Reply in 12 words or fewer. Explain thoroughly with technical detail and examples.",
            "What is a namespace?",
            "word_count_lte", 12, "word_count_gte", 30,
        ),
        (
            "Respond with one sentence.",
            "Respond with one sentence. Ensure you address definitions, examples, and common misconceptions.",
            "What is an exception in programming?",
            "sentence_count", 1, "sentence_count", 4,
        ),
    ]

    for i, (orig, adv, query, cc, cv, ac, av) in enumerate(sc_data):
        pairs.append(ContrastivePair(
            pair_id=f"sc_{i+1:02d}",
            strategy="scope_creep",
            original_instruction=orig,
            adversarial_instruction=adv,
            user_query=query,
            correct_checker=cc,
            correct_value=cv,
            adversarial_checker=ac,
            adversarial_value=av,
        ))

    return pairs


ALL_PAIRS = _build_pairs()


# ── Verdict helpers ───────────────────────────────────────────────────────────

VERDICT_ADHERENT = "adherent"
VERDICT_FOOLED   = "fooled"
VERDICT_PARTIAL  = "partial"


def get_verdict(
    response: str,
    pair: ContrastivePair,
) -> str:
    orig_ok = check_response(response, pair.correct_checker, pair.correct_value)
    adv_ok  = check_response(response, pair.adversarial_checker, pair.adversarial_value)

    if orig_ok and not adv_ok:
        return VERDICT_ADHERENT
    if adv_ok and not orig_ok:
        return VERDICT_FOOLED
    # Both true, both false, or indeterminate → partial
    return VERDICT_PARTIAL


# ── Agent interfaces ──────────────────────────────────────────────────────────

class BaseAgent:
    def generate(self, system_prompt: str, user_message: str) -> str:
        raise NotImplementedError


class DemoAgent(BaseAgent):
    """
    Rule-based agent used in --mode demo.
    Always follows the ORIGINAL instruction perfectly, producing:
      adherence_rate = 100%,  fooling_rate = 0%  on original run
    Also used for the adversarial run — it still follows the original format
    (ignoring the adversarial additions), so fooling_rate stays at 0%.
    """

    def generate(self, system_prompt: str, user_message: str) -> str:
        instr = system_prompt.lower()

        if "json" in instr and "xml" not in instr[:instr.find("json")+4]:
            return '{"answer": "Demo answer"}'
        if "numbered list" in instr or "number each" in instr or "use numbers" in instr:
            return "1. First item\n2. Second item\n3. Third item"
        if "bullet" in instr or "bullet points" in instr:
            return "- First item\n- Second item\n- Third item"
        if "uppercase" in instr or "all caps" in instr or "upper case" in instr:
            return "DEMO ANSWER"
        if "no markdown" in instr or "plain text" in instr or "avoid markdown" in instr:
            return "This is a plain text answer with no special formatting."
        if "one sentence" in instr or "single sentence" in instr or "1 sentence" in instr:
            return "This is a single sentence answer."
        if "two sentence" in instr or "2 sentence" in instr or "exactly 2 sentences" in instr:
            return "This is the first sentence. This is the second sentence."
        if "three sentence" in instr or "3 sentence" in instr or "exactly 3 sentences" in instr:
            return "First sentence here. Second sentence here. Third sentence here."
        if "20 words" in instr or "under 20" in instr:
            return "This concise answer respects the word limit."
        if "15 words" in instr or "under 15" in instr:
            return "Short and precise answer here."
        if "10 words" in instr or "under 10" in instr:
            return "Brief and accurate answer."

        return "Demo answer follows the original instruction."


class HuggingFaceAgent(BaseAgent):
    def __init__(self, model_name: str, max_new_tokens: int = 256):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            print("ERROR: transformers and torch are required for model inference.")
            print("       Install with: pip install transformers torch")
            sys.exit(1)

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        print("Model loaded.")

    def generate(self, system_prompt: str, user_message: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = f"System: {system_prompt}\nUser: {user_message}\nAssistant:"

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── Evaluation loop ───────────────────────────────────────────────────────────

@dataclass
class PairResult:
    pair: ContrastivePair
    original_response: str
    adversarial_response: str
    original_verdict: str
    adversarial_verdict: str
    latency_orig_s: float
    latency_adv_s: float


def evaluate_pair(agent: BaseAgent, pair: ContrastivePair) -> PairResult:
    t0 = time.perf_counter()
    orig_resp = agent.generate(pair.original_instruction, pair.user_query)
    t1 = time.perf_counter()
    adv_resp  = agent.generate(pair.adversarial_instruction, pair.user_query)
    t2 = time.perf_counter()

    return PairResult(
        pair=pair,
        original_response=orig_resp,
        adversarial_response=adv_resp,
        original_verdict=get_verdict(orig_resp, pair),
        adversarial_verdict=get_verdict(adv_resp, pair),
        latency_orig_s=t1 - t0,
        latency_adv_s=t2 - t1,
    )


def run_evaluation(
    agent: BaseAgent,
    pairs: list[ContrastivePair],
    verbose: bool = False,
) -> list[PairResult]:
    results: list[PairResult] = []
    n = len(pairs)
    for idx, pair in enumerate(pairs, 1):
        if verbose:
            print(f"  [{idx:3d}/{n}] {pair.pair_id} ({pair.strategy})", end="", flush=True)
        result = evaluate_pair(agent, pair)
        results.append(result)
        if verbose:
            print(f"  orig={result.original_verdict:8s}  adv={result.adversarial_verdict}")
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

STRATEGIES = [
    "format_override",
    "late_contradiction",
    "implicit_negation",
    "example_hijack",
    "scope_creep",
]


def compute_stats(results: list[PairResult], source: str = "adversarial"):
    """
    source: 'original' or 'adversarial' — which response to score.
    For the main eval we score adversarial responses:
      adherent = model kept original format despite adversarial instruction
      fooled   = model switched to adversarial format
    For --compare we also score original responses.
    """
    rows: dict[str, dict[str, int]] = {s: {"adherent": 0, "fooled": 0, "partial": 0, "n": 0}
                                        for s in STRATEGIES}
    rows["OVERALL"] = {"adherent": 0, "fooled": 0, "partial": 0, "n": 0}

    for r in results:
        verdict = r.adversarial_verdict if source == "adversarial" else r.original_verdict
        s = r.pair.strategy
        rows[s][verdict] += 1
        rows[s]["n"] += 1
        rows["OVERALL"][verdict] += 1
        rows["OVERALL"]["n"] += 1

    return rows


def pct(count: int, total: int) -> str:
    if total == 0:
        return "  N/A"
    return f"{100 * count / total:5.1f}%"


def print_table(rows: dict, title: str):
    print(f"\n=== {title} ===\n")
    header = f"{'Strategy':<22}  {'Adherence':>9}  {'Fooling':>7}  {'Partial':>7}  {'N':>4}"
    print(header)
    print("-" * len(header))
    for s in STRATEGIES:
        r = rows[s]
        print(f"{s:<22}  {pct(r['adherent'], r['n']):>9}  "
              f"{pct(r['fooled'], r['n']):>7}  "
              f"{pct(r['partial'], r['n']):>7}  "
              f"{r['n']:>4}")
    print("-" * len(header))
    r = rows["OVERALL"]
    print(f"{'OVERALL':<22}  {pct(r['adherent'], r['n']):>9}  "
          f"{pct(r['fooled'], r['n']):>7}  "
          f"{pct(r['partial'], r['n']):>7}  "
          f"{r['n']:>4}")
    print()


def print_insights(rows: dict):
    strategy_fooling = {s: rows[s]["fooled"] / rows[s]["n"]
                        for s in STRATEGIES if rows[s]["n"] > 0}
    strategy_adherence = {s: rows[s]["adherent"] / rows[s]["n"]
                          for s in STRATEGIES if rows[s]["n"] > 0}

    most_vulnerable = max(strategy_fooling, key=strategy_fooling.get)
    most_robust     = max(strategy_adherence, key=strategy_adherence.get)

    print(f"Most vulnerable strategy : {most_vulnerable} "
          f"({100*strategy_fooling[most_vulnerable]:.1f}% fooling rate)")
    print(f"Most robust strategy     : {most_robust} "
          f"({100*strategy_adherence[most_robust]:.1f}% adherence)")

    # Two worst strategies and their share of total failures
    total_fooled = rows["OVERALL"]["fooled"]
    if total_fooled > 0:
        sorted_by_fooled = sorted(STRATEGIES, key=lambda s: rows[s]["fooled"], reverse=True)
        top2 = sorted_by_fooled[:2]
        top2_fooled = sum(rows[s]["fooled"] for s in top2)
        share = 100 * top2_fooled / total_fooled
        print(f"\nKey finding: {top2[0]} + {top2[1]} account for "
              f"{share:.0f}% of all failures.")
        if top2[0] in ("late_contradiction", "scope_creep") or \
           top2[1] in ("late_contradiction", "scope_creep"):
            print("Instruction stacking (adding constraints that contradict earlier ones) "
                  "is the dominant failure mode.")


def print_compare(orig_rows: dict, adv_rows: dict):
    print("\n=== Side-by-Side: Original vs Adversarial Instruction ===\n")
    header = (f"{'Strategy':<22}  "
              f"{'Orig Adhere':>11}  {'Adv Adhere':>10}  "
              f"{'Orig Fool':>9}  {'Adv Fool':>8}  {'N':>4}")
    print(header)
    print("-" * len(header))
    for s in STRATEGIES:
        o, a = orig_rows[s], adv_rows[s]
        print(f"{s:<22}  "
              f"{pct(o['adherent'], o['n']):>11}  {pct(a['adherent'], a['n']):>10}  "
              f"{pct(o['fooled'], o['n']):>9}  {pct(a['fooled'], a['n']):>8}  "
              f"{o['n']:>4}")
    print("-" * len(header))
    o, a = orig_rows["OVERALL"], adv_rows["OVERALL"]
    print(f"{'OVERALL':<22}  "
          f"{pct(o['adherent'], o['n']):>11}  {pct(a['adherent'], a['n']):>10}  "
          f"{pct(o['fooled'], o['n']):>9}  {pct(a['fooled'], a['n']):>8}  "
          f"{o['n']:>4}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Contrastive instruction robustness evaluator")
    ap.add_argument("--mode", choices=["demo", "model"], default="demo",
                    help="'demo' uses a rule-based agent; 'model' loads a HuggingFace model")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="HuggingFace model name (used when --mode model)")
    ap.add_argument("--n", type=int, default=200,
                    help="Number of pairs to evaluate (max 200)")
    ap.add_argument("--strategy", choices=STRATEGIES + ["all"], default="all",
                    help="Run only a specific strategy")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-pair results during evaluation")
    ap.add_argument("--compare", action="store_true",
                    help="Show side-by-side original vs adversarial performance")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="Max tokens for model generation (model mode only)")
    return ap.parse_args()


def main():
    args = parse_args()

    # Filter pairs
    pairs = ALL_PAIRS
    if args.strategy != "all":
        pairs = [p for p in pairs if p.strategy == args.strategy]
    if args.n < len(pairs):
        pairs = pairs[: args.n]

    print(f"Evaluating {len(pairs)} contrastive pairs "
          f"(mode={args.mode}, strategy={args.strategy})")

    # Build agent
    if args.mode == "demo":
        agent = DemoAgent()
    else:
        agent = HuggingFaceAgent(args.model, max_new_tokens=args.max_new_tokens)

    # Run eval
    results = run_evaluation(agent, pairs, verbose=args.verbose)

    # Stats: adversarial responses (primary metric)
    adv_rows  = compute_stats(results, source="adversarial")
    orig_rows = compute_stats(results, source="original")

    n_total = len(pairs)
    print_table(adv_rows,
                f"Contrastive Instruction Robustness ({n_total} pairs)")
    print_insights(adv_rows)

    if args.compare:
        print_compare(orig_rows, adv_rows)

    # Per-pair failures summary
    failures = [r for r in results if r.adversarial_verdict == VERDICT_FOOLED]
    if failures and args.verbose:
        print(f"\n--- Failures ({len(failures)}) ---")
        for r in failures[:10]:
            print(f"  {r.pair.pair_id}: {r.pair.strategy}")
            print(f"    Q: {r.pair.user_query[:60]}")
            print(f"    orig_instr : {r.pair.original_instruction[:60]}")
            print(f"    adv_instr  : {r.pair.adversarial_instruction[:60]}")
            print(f"    adv_resp   : {r.adversarial_response[:80]}")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
