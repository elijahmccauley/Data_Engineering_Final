"""
modules/description_generation.py — Description Generation Module.

Generates three flavours of dataset description, in increasing richness:
  1. rule_based   — Pure template, no LLM.  Fast, reproducible, zero cost.
  2. tabular_only — LLM with structured profile only (AutoDDG baseline style).
  3. tabular_text — LLM with profile + semantic text summary (our extension).

Design decisions:
  - Each generator is a standalone function so teammates can import and call
    individual variants without running the whole pipeline.
  - Prompts explicitly forbid hallucination ("only use information above") and
    use a structured instruction list to guide the LLM's output format.
  - The rule-based generator is intentionally minimal — it serves as the
    no-LLM comparison point in the evaluation.
"""

from __future__ import annotations

import json

import pandas as pd
from openai import OpenAI

import config
from utils.openai_utils import call_openai


# ── 1. Rule-based description (no LLM) ────────────────────────────────────────

def generate_rule_based_description(df: pd.DataFrame) -> str:
    """
    Build a dataset description from hard-coded string templates and
    value-count statistics. No LLM call — entirely deterministic.

    Assumes the DataFrame has the standard fashion dataset columns.
    Falls back gracefully if an expected column is missing.
    """
    num_rows, num_cols = df.shape

    def top_values(col: str, k: int = 5) -> str:
        if col not in df.columns:
            return "N/A"
        return ", ".join(df[col].value_counts().head(k).index.tolist())

    genders = top_values("Gender", k=10)
    categories = top_values("Category", k=10)
    subcategories = top_values("SubCategory")
    product_types = top_values("ProductType")
    colours = top_values("Colour")
    usages = top_values("Usage")

    missing = df.isna().sum()
    has_missing = missing[missing > 0]
    missing_note = (
        "No missing values were detected."
        if has_missing.empty
        else f"Missing values found in: {', '.join(has_missing.index.tolist())}."
    )

    return (
        f"This dataset contains {num_rows} e-commerce fashion product records "
        f"across {num_cols} columns.\n"
        f"It includes structured product metadata such as gender, category, subcategory, "
        f"product type, colour, usage, product title, image filename, and image URL.\n\n"
        f"The dataset primarily covers {categories} products for {genders}. "
        f"The most common subcategories include {subcategories}, and the most frequent "
        f"product types include {product_types}. "
        f"Common colours include {colours}, while the dominant usage context is {usages}.\n\n"
        f"{missing_note}\n\n"
        f"The dataset can support product categorisation, metadata analysis, recommendation, "
        f"and multimodal learning tasks involving product attributes, titles, and images."
    )


# ── 2. LLM — tabular-only description ─────────────────────────────────────────

def _build_tabular_only_prompt(dataset_name: str, compact_profile: dict) -> str:
    return f"""You are generating a dataset description for a data catalogue.

Dataset name: {dataset_name}

Structured profile:
{json.dumps(compact_profile, indent=2)}

Write a clear, informative description that covers:
1. What the dataset is about and its domain.
2. What entities or items it contains.
3. Its main attributes, column types, and any notable statistics.
4. What tasks or analyses it can support.

Rules:
- Only use information present in the profile above.
- Do not invent fields, brands, or details not listed.
- Be concise but specific — avoid generic filler sentences.
- Length: 150–250 words."""


def generate_tabular_only_description(
    dataset_name: str,
    compact_profile: dict,
    client: OpenAI | None = None,
    model: str = config.DEFAULT_MODEL,
) -> str:
    """
    Generate a description using only the structured tabular profile (no text).
    This is the AutoDDG baseline approach.
    """
    prompt = _build_tabular_only_prompt(dataset_name, compact_profile)
    return call_openai(
        prompt=prompt,
        system_message=(
            "You generate high-quality dataset descriptions from structured data summaries. "
            "Never hallucinate. Only describe what is shown in the profile."
        ),
        model=model,
        temperature=0.3,
        client=client,
    )


# ── 3. LLM — tabular + text description ───────────────────────────────────────

def _build_tabular_text_prompt(
    dataset_name: str,
    compact_profile: dict,
    text_semantic_summary: str,
    text_samples: dict[str, list[str]],
) -> str:
    sample_lines = []
    for col, vals in text_samples.items():
        sample_lines.append(f"Column '{col}' samples:")
        sample_lines += [f"  - {v}" for v in vals[:10]]

    return f"""You are generating a dataset description for a data catalogue.

Dataset name: {dataset_name}

Structured profile:
{json.dumps(compact_profile, indent=2)}

Semantic summary of text fields (LLM-generated):
{text_semantic_summary}

Sample text values:
{chr(10).join(sample_lines)}

Write a clear, informative description that covers:
1. What the dataset is about and its domain.
2. What entities or items it contains (use real examples from the samples).
3. Its main attributes, structure, and statistics.
4. Semantic insights drawn from the text fields (themes, brands, styles).
5. What tasks or analyses it can support.

Rules:
- Only use information present in the profile and samples above.
- Do not invent brands, fields, or statistics not listed.
- Incorporate specific product examples where relevant.
- Be concise but richer than a pure tabular description.
- Length: 200–300 words."""


def generate_tabular_text_description(
    dataset_name: str,
    compact_profile: dict,
    text_semantic_summary: str,
    text_samples: dict[str, list[str]],
    client: OpenAI | None = None,
    model: str = config.DEFAULT_MODEL,
) -> str:
    """
    Generate an enriched description that combines the tabular profile with
    semantic insights extracted from the dataset's text columns.

    This is the text-enhanced variant.
    """
    prompt = _build_tabular_text_prompt(
        dataset_name, compact_profile, text_semantic_summary, text_samples
    )
    return call_openai(
        prompt=prompt,
        system_message=(
            "You generate high-quality dataset descriptions that combine structured "
            "tabular information with semantic text insights. Be factual and specific."
        ),
        model=model,
        temperature=0.3,
        client=client,
    )
