"""
modules/text_processing.py — Text Processing Module.

Responsibility:
  1. Detect which columns carry meaningful natural-language content
     (semantic text columns), excluding noise like URLs and image filenames.
  2. Sample a representative subset of values from those columns.
  3. Call an LLM to produce a short semantic summary of the text content.

Design decisions:
  - Detection uses two complementary signals: average string length and
    keyword hints in the column name (title, name, description). Using both
    avoids missing short-but-meaningful columns and catches URL-like columns
    that happen to be long.
  - The exclusion list is keyword-based (url, imageurl, image filename) rather
    than an exact match, so it generalises across dataset schemas.
  - Sampling is deterministic (fixed random_state) for reproducibility.
  - The LLM prompt is tightly scoped: it asks only for a semantic summary,
    not a full description, to keep concerns separate.
"""

from __future__ import annotations

import pandas as pd
from openai import OpenAI

import config
from utils.openai_utils import call_openai


# ── Column patterns to exclude from semantic text detection ────────────────────
_EXCLUDE_PATTERNS = ("url", "imageurl", "image")


def _is_excluded(col_name: str) -> bool:
    """Return True if the column name matches a known non-semantic pattern."""
    lower = col_name.lower()
    return any(p in lower for p in _EXCLUDE_PATTERNS)


def _is_filename(series: pd.Series, sample_n: int = 50) -> bool:
    """
    Heuristic: if most sampled values end with a known file extension, treat
    the column as a filename column (not semantic text).
    """
    FILE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".csv", ".json")
    sample = series.dropna().astype(str).head(sample_n)
    if sample.empty:
        return False
    hits = sample.apply(lambda v: v.lower().endswith(FILE_EXTS)).sum()
    return (hits / len(sample)) > 0.5


# ── Public API ─────────────────────────────────────────────────────────────────

def detect_semantic_text_columns(
    df: pd.DataFrame,
    min_avg_length: int = config.TEXT_MIN_AVG_LENGTH,
) -> list[str]:
    """
    Return a list of column names that likely contain semantic natural-language
    text (e.g. product titles, descriptions, review text).

    A column qualifies if it is object-typed AND:
      - is NOT on the exclusion list (url, image …), AND
      - is NOT mostly file paths, AND
      - EITHER its average string length ≥ min_avg_length
        OR its name contains a semantic hint (title, name, description, text).
    """
    semantic_cols = []

    for col in df.columns:
        if df[col].dtype != "object":
            continue
        if _is_excluded(col):
            continue

        values = df[col].dropna().astype(str)
        if values.empty:
            continue

        if _is_filename(values):
            continue

        col_lower = col.lower()
        name_hint = any(
            kw in col_lower for kw in ("title", "name", "description", "text", "review", "comment")
        )
        avg_len = values.str.len().mean()

        if avg_len >= min_avg_length or name_hint:
            semantic_cols.append(col)

    return semantic_cols


def sample_semantic_text(
    df: pd.DataFrame,
    cols: list[str],
    max_samples: int = config.TEXT_MAX_SAMPLES,
    random_state: int = config.TEXT_SAMPLE_RANDOM_STATE,
) -> dict[str, list[str]]:
    """
    Sample up to max_samples unique, non-null values from each column in cols.

    Returns a dict {column_name: [value, ...]}.
    Sampling is deterministic via random_state for reproducibility.
    """
    samples: dict[str, list[str]] = {}

    for col in cols:
        unique_vals = df[col].dropna().astype(str).drop_duplicates()
        n = min(max_samples, len(unique_vals))
        if n == 0:
            continue
        samples[col] = unique_vals.sample(n, random_state=random_state).tolist()

    return samples


def generate_text_semantic_summary(
    dataset_name: str,
    text_samples: dict[str, list[str]],
    client: OpenAI | None = None,
    model: str = config.DEFAULT_MODEL,
) -> str:
    """
    Call the LLM to produce a short semantic summary of the sampled text values.

    The prompt focuses on:
      - types of items/products mentioned
      - common themes, attributes, brand names
      - the domain
      - potential analytical tasks

    Args:
        dataset_name: Human-readable dataset name (for context).
        text_samples: Dict returned by sample_semantic_text.
        client: Optional pre-built OpenAI client.
        model: LLM model to use.

    Returns:
        A plain-text semantic summary string.
    """
    lines = []
    for col, vals in text_samples.items():
        lines.append(f"Column: {col}")
        lines += [f"  - {v}" for v in vals]
        lines.append("")

    prompt = f"""You are analysing a dataset and summarising the semantic meaning of its text fields.

Dataset name: {dataset_name}

Sampled text values:
{chr(10).join(lines)}

Write a concise semantic summary (3-5 sentences) covering:
1. What kinds of products or items appear in the data.
2. Common themes, attributes, or brand names.
3. The domain of the dataset.
4. What analytical tasks or ML tasks the text could support.

Do NOT invent information not visible in the samples."""

    return call_openai(
        prompt=prompt,
        system_message=(
            "You are a careful data analyst who summarises the semantic content "
            "of datasets. Be factual and concise."
        ),
        model=model,
        temperature=0.2,
        client=client,
    )
