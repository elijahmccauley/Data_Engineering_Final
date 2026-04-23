"""
modules/profiling.py — Dataset Profiling Module.

Responsibility: Given a pandas DataFrame, produce a structured profile dict
that can be fed into LLM prompts or inspection tooling.

Design decisions:
  - Numeric and categorical columns are handled separately.
  - The "compact" profile limits columns to those relevant for LLM prompts,
    avoiding unnecessarily long context windows.
  - All numeric outputs are cast to Python-native types (float/int) so the
    profile is immediately JSON-serialisable.
"""

from __future__ import annotations

import pandas as pd
import config


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _numeric_summary(series: pd.Series) -> dict:
    """Return min/max/mean/median for a numeric-like column."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": round(float(s.mean()), 4),
        "median": float(s.median()),
    }


def _top_values(series: pd.Series, top_k: int = config.TOP_K_VALUES) -> dict:
    """Return the top-k value counts as a plain {value: count} dict."""
    vc = series.astype(str).value_counts(dropna=False).head(top_k)
    return {str(k): int(v) for k, v in vc.items()}


# ── Core profiling ─────────────────────────────────────────────────────────────

def build_dataset_profile(df: pd.DataFrame) -> dict:
    """
    Build a full profile of every column in the DataFrame.

    Returns a dict with:
      - num_rows, num_columns, columns
      - column_types: {col: dtype string}
      - missing_values: {col: int count}
      - column_profiles: per-column detail (top_values or numeric_summary)
    """
    profile = {
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "column_types": df.dtypes.astype(str).to_dict(),
        "missing_values": {col: int(n) for col, n in df.isna().sum().items()},
        "column_profiles": {},
    }

    for col in df.columns:
        col_data = df[col]
        entry = {
            "dtype": str(col_data.dtype),
            "missing_count": int(col_data.isna().sum()),
            "unique_count": int(col_data.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(col_data):
            entry["numeric_summary"] = _numeric_summary(col_data)
        else:
            entry["top_values"] = _top_values(col_data)

        profile["column_profiles"][col] = entry

    return profile


def build_compact_profile(
    profile: dict,
    important_columns: list[str] | None = None,
) -> dict:
    """
    Return a slimmed-down profile containing only the columns that matter for
    LLM prompts. This prevents prompt bloat when the dataset has many columns.

    Args:
        profile: Full profile returned by build_dataset_profile.
        important_columns: Columns to retain. Defaults to config setting.
    """
    if important_columns is None:
        important_columns = config.IMPORTANT_PROFILE_COLUMNS

    return {
        "num_rows": profile["num_rows"],
        "num_columns": profile["num_columns"],
        "columns": profile["columns"],
        "column_types": profile["column_types"],
        "selected_column_profiles": {
            col: profile["column_profiles"][col]
            for col in important_columns
            if col in profile["column_profiles"]
        },
    }


def print_profile_summary(profile: dict) -> None:
    """Pretty-print a quick human-readable summary of the profile."""
    print(f"Rows: {profile['num_rows']}  |  Columns: {profile['num_columns']}")
    print("\nMissing values:")
    for col, n in profile["missing_values"].items():
        flag = "  ← has missing!" if n > 0 else ""
        print(f"  {col}: {n}{flag}")
    print("\nColumn profiles:")
    for col, info in profile["column_profiles"].items():
        print(f"  [{info['dtype']}] {col}  ({info['unique_count']} unique)")
        if "top_values" in info:
            top = list(info["top_values"].items())[:5]
            print("    top:", ", ".join(f"{k}({v})" for k, v in top))
        elif "numeric_summary" in info:
            ns = info["numeric_summary"]
            print(f"    range: {ns.get('min')} – {ns.get('max')}  mean: {ns.get('mean')}")
