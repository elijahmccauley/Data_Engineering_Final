"""
modules/evaluation.py — Evaluation Module (Yuheng's core contribution).

Three complementary evaluation methods for comparing dataset descriptions:

  A. Pointwise  — LLM scores each description on 4 criteria (1-5 scale).
                  Uses calibrated anchors so scores are not all 5s.
  B. Pairwise   — LLM picks the better description for dataset discovery
                  between two candidates, with reasoning.
  C. Question   — Generates dataset-specific questions; checks how many
                  each description answers, producing a coverage score.

Design decisions:
  - Each method returns a plain Python dict so the caller (main.py) can
    decide how to serialise (CSV, JSON, etc.).
  - Temperature is set to 0 for all evaluation calls — we want deterministic
    judgements, not creative variation.
  - STRICT scoring rubrics are embedded in every prompt to prevent the
    well-known LLM tendency to award uniformly high scores.
  - The question-based method generates questions from the dataset profile,
    not hard-coded ones, so it generalises to other datasets.
"""

from __future__ import annotations

import json
import re

from openai import OpenAI

import config
from utils.openai_utils import call_openai


# ── A. Pointwise Evaluation ────────────────────────────────────────────────────

_POINTWISE_SYSTEM = """You are a strict, critical evaluator of dataset descriptions.
Your job is to score descriptions honestly.

IMPORTANT SCORING RULES:
- A score of 5 means near-perfect with essentially nothing missing.
- A score of 4 means good but with at least one noticeable gap.
- A score of 3 means adequate but missing several important details.
- A score of 2 means poor — major gaps or confusing language.
- A score of 1 means very poor or essentially useless.

Do NOT give 5s unless truly warranted. Prefer 3 or 4 for decent descriptions.
You MUST return valid JSON only. No markdown, no extra text."""


def _pointwise_prompt(
    dataset_name: str,
    compact_profile: dict,
    description_name: str,
    description_text: str,
) -> str:
    return f"""Evaluate the following dataset description for quality.

Dataset name: {dataset_name}

Reference dataset profile (ground truth):
{json.dumps(compact_profile, indent=2)}

Description type: {description_name}

Description text:
\"\"\"{description_text}\"\"\"

Score each dimension from 1 (very poor) to 5 (excellent):

1. completeness  — Does it cover all key aspects: domain, entities, attributes, tasks?
                   Deduct points for missing columns, skipped statistics, or omitted tasks.
2. clarity       — Is it easy to read and well-structured?
                   Deduct for jargon, run-on sentences, or poor organisation.
3. specificity   — Does it use concrete details (real column names, actual counts, specific examples)?
                   Deduct heavily for vague or generic statements.
4. usefulness    — Would a user finding this description be able to quickly judge dataset fit?
                   Deduct if key details for discovery are buried or absent.

Return ONLY valid JSON in this exact format:
{{
  "completeness": <1-5>,
  "clarity": <1-5>,
  "specificity": <1-5>,
  "usefulness": <1-5>,
  "overall": <average rounded to 1 decimal>,
  "strengths": "<one sentence>",
  "weaknesses": "<one sentence>"
}}"""


def evaluate_pointwise(
    dataset_name: str,
    compact_profile: dict,
    descriptions: dict[str, str],
    client: OpenAI | None = None,
    model: str = config.DEFAULT_MODEL,
) -> dict[str, dict]:
    """
    Score each description individually on completeness, clarity, specificity,
    and usefulness.

    Args:
        descriptions: {method_name: description_text} mapping.

    Returns:
        {method_name: {completeness, clarity, specificity, usefulness, overall,
                        strengths, weaknesses}}
    """
    results = {}

    for name, text in descriptions.items():
        prompt = _pointwise_prompt(dataset_name, compact_profile, name, text)
        raw = call_openai(
            prompt=prompt,
            system_message=_POINTWISE_SYSTEM,
            model=model,
            temperature=0,
            client=client,
        )

        # Strip markdown fences if the model wraps output
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        try:
            results[name] = json.loads(clean)
        except json.JSONDecodeError:
            results[name] = {"raw_output": raw, "parse_error": True}

    return results


# ── B. Pairwise Comparison ─────────────────────────────────────────────────────

_PAIRWISE_SYSTEM = """You are an expert judge comparing dataset descriptions for discoverability.
Be decisive — always pick a winner. Return valid JSON only."""


def _pairwise_prompt(
    dataset_name: str,
    compact_profile: dict,
    name_a: str,
    text_a: str,
    name_b: str,
    text_b: str,
) -> str:
    return f"""You are comparing two dataset descriptions to decide which is more useful
for someone trying to discover and understand a dataset.

Dataset name: {dataset_name}

Reference dataset profile:
{json.dumps(compact_profile, indent=2)}

--- Description A ({name_a}) ---
{text_a}

--- Description B ({name_b}) ---
{text_b}

Evaluate along these axes:
- Which conveys more about the dataset's actual content?
- Which would help a researcher decide faster whether the dataset suits their needs?
- Which provides more concrete, actionable information?

Return ONLY valid JSON:
{{
  "winner": "{name_a}" | "{name_b}" | "tie",
  "margin": "clear" | "slight",
  "reasoning": "<2-3 sentences explaining the decision>",
  "a_advantage": "<what A does better, or 'nothing'>",
  "b_advantage": "<what B does better, or 'nothing'>"
}}"""


def evaluate_pairwise(
    dataset_name: str,
    compact_profile: dict,
    comparisons: list[tuple[str, str, str, str]],
    client: OpenAI | None = None,
    model: str = config.DEFAULT_MODEL,
) -> list[dict]:
    """
    Run one or more head-to-head comparisons between descriptions.

    Args:
        comparisons: List of (name_a, text_a, name_b, text_b) tuples.

    Returns:
        List of result dicts, one per comparison, each containing
        {name_a, name_b, winner, margin, reasoning, a_advantage, b_advantage}.
    """
    results = []

    for name_a, text_a, name_b, text_b in comparisons:
        prompt = _pairwise_prompt(
            dataset_name, compact_profile, name_a, text_a, name_b, text_b
        )
        raw = call_openai(
            prompt=prompt,
            system_message=_PAIRWISE_SYSTEM,
            model=model,
            temperature=0,
            client=client,
        )

        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        try:
            result = json.loads(clean)
        except json.JSONDecodeError:
            result = {"raw_output": raw, "parse_error": True}

        result["name_a"] = name_a
        result["name_b"] = name_b
        results.append(result)

    return results


# ── C. Question-based Evaluation ───────────────────────────────────────────────

_QUESTION_GEN_SYSTEM = """You generate concise, answerable evaluation questions about datasets.
Return valid JSON only."""

_ANSWER_CHECK_SYSTEM = """You check whether a dataset description answers specific questions.
Be strict: only mark a question as answered if the description directly addresses it.
Return valid JSON only."""


def generate_evaluation_questions(
    dataset_name: str,
    compact_profile: dict,
    n_questions: int = 7,
    client: OpenAI | None = None,
    model: str = config.DEFAULT_MODEL,
) -> list[str]:
    """
    Ask the LLM to generate dataset-specific evaluation questions based on
    the profile. Questions are grounded in the actual data, not generic.

    Returns a list of question strings.
    """
    prompt = f"""You are preparing evaluation questions for a dataset description.

Dataset name: {dataset_name}
Profile summary:
  - Rows: {compact_profile['num_rows']}
  - Columns: {compact_profile['columns']}
  - Column types: {compact_profile['column_types']}

Generate exactly {n_questions} specific questions that a good dataset description
should be able to answer. Make them concrete and tied to this dataset.

Include questions of these types:
  - domain/topic (What is this dataset about?)
  - entity (What are the main items or records?)
  - attribute (What columns or fields are available?)
  - statistics (How many records / what range of values?)
  - task (What ML or analytical tasks can this support?)
  - coverage (What demographic groups or categories are present?)
  - quality (Are there missing values or known limitations?)

Return ONLY valid JSON — a list of strings:
["question 1", "question 2", ...]"""

    raw = call_openai(
        prompt=prompt,
        system_message=_QUESTION_GEN_SYSTEM,
        model=model,
        temperature=0.3,
        client=client,
    )
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        questions = json.loads(clean)
        return questions if isinstance(questions, list) else []
    except json.JSONDecodeError:
        return []


def _check_answers_prompt(
    description_name: str,
    description_text: str,
    questions: list[str],
) -> str:
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    return f"""A dataset description is shown below. For each question, decide whether
the description directly answers it (true) or does not (false).

Description type: {description_name}

Description:
\"\"\"{description_text}\"\"\"

Questions:
{numbered}

Be strict: only mark true if the answer is explicitly present or strongly implied.

Return ONLY valid JSON — a list of objects in the same order as the questions:
[
  {{"question": "...", "answered": true | false, "evidence": "<quoted phrase or 'none'>"}},
  ...
]"""


def evaluate_question_based(
    descriptions: dict[str, str],
    questions: list[str],
    client: OpenAI | None = None,
    model: str = config.DEFAULT_MODEL,
) -> dict[str, dict]:
    """
    For each description, check how many of the evaluation questions it answers.

    Returns a dict:
    {
      method_name: {
        "answered": int,
        "total": int,
        "coverage": float (0-1),
        "details": [{question, answered, evidence}, ...]
      }
    }
    """
    results = {}

    for name, text in descriptions.items():
        prompt = _check_answers_prompt(name, text, questions)
        raw = call_openai(
            prompt=prompt,
            system_message=_ANSWER_CHECK_SYSTEM,
            model=model,
            temperature=0,
            client=client,
        )
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)

        try:
            details = json.loads(clean)
            if not isinstance(details, list):
                raise ValueError("Expected a list")
            answered = sum(1 for d in details if d.get("answered") is True)
            results[name] = {
                "answered": answered,
                "total": len(questions),
                "coverage": round(answered / len(questions), 3) if questions else 0.0,
                "details": details,
            }
        except (json.JSONDecodeError, ValueError):
            results[name] = {"raw_output": raw, "parse_error": True}

    return results
