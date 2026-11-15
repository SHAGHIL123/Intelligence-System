"""Load, clean, and profile user-uploaded CSV and Excel data."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO

import numpy as np
import pandas as pd


_IDENTIFIER_NAME_PARTS = ("id", "code", "zip", "postal", "phone")
_MAX_PROFILE_COLUMNS = 30
_MAX_CORRELATIONS = 20


@dataclass
class AnalysisResult:
    """Cleaned data and analysis material suitable for an LLM prompt."""

    dataframe: pd.DataFrame
    cleaning_report: dict[str, object]
    eda_report: dict[str, object]
    llm_summary: str


def _read_upload(uploaded_file: BinaryIO, filename: str) -> pd.DataFrame:
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    content = uploaded_file.read()
    if not content:
        raise ValueError("The uploaded file is empty.")

    if suffix == "csv":
        try:
            return pd.read_csv(BytesIO(content), encoding="utf-8-sig", dtype="object")
        except UnicodeDecodeError:
            return pd.read_csv(BytesIO(content), encoding="cp1252", dtype="object")
        except pd.errors.EmptyDataError as error:
            raise ValueError("The CSV file does not contain tabular data.") from error
    if suffix == "xlsx":
        try:
            return pd.read_excel(BytesIO(content), engine="openpyxl")
        except pd.errors.EmptyDataError as error:
            raise ValueError("The Excel file does not contain tabular data.") from error

    raise ValueError("Unsupported file type. Upload a .csv or .xlsx file.")


def _clean_column_names(columns: pd.Index) -> tuple[list[str], list[dict[str, str]]]:
    cleaned: list[str] = []
    changes: list[dict[str, str]] = []
    used: set[str] = set()

    for index, original in enumerate(columns):
        source = str(original).strip()
        base = re.sub(r"\W+", "_", source.lower(), flags=re.UNICODE).strip("_")
        if not base:
            base = f"column_{index + 1}"

        name = base
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        cleaned.append(name)

        if name != source:
            changes.append({"from": source, "to": name})

    return cleaned, changes


def _convert_column_types(dataframe: pd.DataFrame) -> list[dict[str, str]]:
    conversions: list[dict[str, str]] = []

    for column in dataframe.columns:
        series = dataframe[column]
        old_type = str(series.dtype)

        if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
            normalized = series.map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
            normalized = normalized.replace("", pd.NA)
            dataframe[column] = normalized
            series = dataframe[column]

        if not (pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype)):
            continue

        name_parts = set(column.split("_"))
        is_identifier = any(
            part in name_parts or column.endswith(f"_{part}")
            for part in _IDENTIFIER_NAME_PARTS
        )
        non_missing = series.dropna()
        if non_missing.empty or is_identifier:
            continue

        numeric_text = non_missing.astype("string").str.replace(
            r"[$£€,]", "", regex=True
        ).str.replace("%", "", regex=False)
        numeric = pd.to_numeric(numeric_text, errors="coerce")
        if numeric.notna().mean() >= 0.9:
            converted = series.astype("string").str.strip()
            converted = converted.str.replace(r"[$£€,]", "", regex=True).str.replace(
                "%", "", regex=False
            )
            dataframe[column] = pd.to_numeric(converted, errors="coerce")
            new_type = str(dataframe[column].dtype)
            if new_type != old_type:
                conversions.append({"column": column, "from": old_type, "to": new_type})
            continue

        looks_like_date = any(
            token in column for token in ("date", "time", "timestamp")
        )
        if looks_like_date:
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().mean() >= 0.8:
                dataframe[column] = parsed
                conversions.append(
                    {"column": column, "from": old_type, "to": str(parsed.dtype)}
                )

    return conversions


def _python_value(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _number(value: object) -> float | int | None:
    normalized = _python_value(value)
    if normalized is None:
        return None
    if isinstance(normalized, (int, float)):
        return normalized
    return float(normalized)


def _build_eda_report(dataframe: pd.DataFrame) -> dict[str, object]:
    row_count, column_count = dataframe.shape
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    datetime_columns = dataframe.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical_columns = [
        column
        for column in dataframe.columns
        if column not in numeric_columns and column not in datetime_columns
    ]

    missing_by_column = {
        column: {
            "count": int(dataframe[column].isna().sum()),
            "percent": round(float(dataframe[column].isna().mean() * 100), 2)
            if row_count
            else 0.0,
        }
        for column in dataframe.columns
    }

    numeric_metrics: dict[str, dict[str, object]] = {}
    for column in numeric_columns[:_MAX_PROFILE_COLUMNS]:
        values = dataframe[column].replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        outlier_count = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())

        numeric_metrics[column] = {
            "count": int(values.count()),
            "distinct_count": int(values.nunique()),
            "sum": _number(values.sum()),
            "mean": _number(values.mean()),
            "median": _number(values.median()),
            "std": _number(values.std()),
            "min": _number(values.min()),
            "q1": _number(q1),
            "q3": _number(q3),
            "max": _number(values.max()),
            "zero_count": int((values == 0).sum()),
            "outlier_count_iqr": outlier_count,
        }

    categorical_metrics: dict[str, dict[str, object]] = {}
    for column in categorical_columns[:_MAX_PROFILE_COLUMNS]:
        values = dataframe[column].dropna()
        counts = values.value_counts(dropna=True)
        categorical_metrics[column] = {
            "non_missing_count": int(values.size),
            "distinct_count": int(values.nunique()),
            "top_value_count": int(counts.iloc[0]) if not counts.empty else 0,
            "top_value_percent": round(float(counts.iloc[0] / values.size * 100), 2)
            if values.size and not counts.empty
            else 0.0,
        }

    datetime_metrics: dict[str, dict[str, object]] = {}
    for column in datetime_columns[:_MAX_PROFILE_COLUMNS]:
        values = dataframe[column].dropna()
        datetime_metrics[column] = {
            "non_missing_count": int(values.size),
            "min": values.min().isoformat() if not values.empty else None,
            "max": values.max().isoformat() if not values.empty else None,
        }

    correlations: list[dict[str, object]] = []
    profiled_numeric_columns = list(numeric_metrics)
    if len(profiled_numeric_columns) > 1:
        corr = dataframe[profiled_numeric_columns].corr(numeric_only=True)
        for index, column in enumerate(profiled_numeric_columns):
            for other in profiled_numeric_columns[index + 1 :]:
                coefficient = corr.loc[column, other]
                if pd.notna(coefficient) and abs(coefficient) >= 0.7:
                    correlations.append(
                        {
                            "column_a": column,
                            "column_b": other,
                            "correlation": round(float(coefficient), 3),
                        }
                    )
        correlations.sort(key=lambda pair: abs(float(pair["correlation"])), reverse=True)

    highlights: list[str] = []
    if row_count and column_count:
        highlights.append(f"The cleaned dataset contains {row_count:,} rows and {column_count:,} columns.")
    if numeric_metrics:
        most_missing = max(missing_by_column, key=lambda column: missing_by_column[column]["percent"])
        if missing_by_column[most_missing]["count"]:
            highlights.append(
                f"'{most_missing}' has the highest missingness "
                f"({missing_by_column[most_missing]['percent']}%)."
            )
        outlier_columns = [
            column
            for column, metrics in numeric_metrics.items()
            if metrics["outlier_count_iqr"]
        ]
        if outlier_columns:
            highlights.append(
                f"IQR outliers were detected in {len(outlier_columns)} numeric column(s)."
            )
    if correlations:
        strongest = correlations[0]
        highlights.append(
            f"The strongest high correlation is between '{strongest['column_a']}' and "
            f"'{strongest['column_b']}' (r={strongest['correlation']})."
        )

    return {
        "dataset": {
            "rows": row_count,
            "columns": column_count,
            "numeric_columns": len(numeric_columns),
            "categorical_columns": len(categorical_columns),
            "datetime_columns": len(datetime_columns),
        },
        "missing_values": {
            "total_cells": row_count * column_count,
            "missing_cells": int(dataframe.isna().sum().sum()),
            "missing_percent": round(float(dataframe.isna().to_numpy().mean() * 100), 2)
            if row_count and column_count
            else 0.0,
            "by_column": missing_by_column,
        },
        "numeric_metrics": numeric_metrics,
        "categorical_metrics": categorical_metrics,
        "datetime_metrics": datetime_metrics,
        "high_correlations": correlations[:_MAX_CORRELATIONS],
        "highlights": highlights,
        "profile_limits": {
            "max_columns_per_type": _MAX_PROFILE_COLUMNS,
            "max_correlations": _MAX_CORRELATIONS,
        },
    }


def analyze_upload(
    uploaded_file: BinaryIO,
    filename: str | None = None,
) -> AnalysisResult:
    """Clean an uploaded CSV/XLSX file and prepare EDA and LLM-ready context.

    Partial missing values are preserved rather than imputed. Fully empty rows
    and columns and duplicate rows are removed; detected outliers are reported,
    not discarded. The summary contains aggregate metrics, not dataset rows.
    """
    resolved_filename = filename or getattr(uploaded_file, "name", None)
    if not resolved_filename:
        raise ValueError("A filename is required to determine the uploaded file type.")

    dataframe = _read_upload(uploaded_file, resolved_filename)
    if dataframe.shape[1] == 0:
        raise ValueError("The uploaded file does not contain any columns.")

    original_rows, original_columns = dataframe.shape
    dataframe.columns, renamed_columns = _clean_column_names(dataframe.columns)

    for column in dataframe.columns:
        if pd.api.types.is_object_dtype(dataframe[column].dtype) or pd.api.types.is_string_dtype(
            dataframe[column].dtype
        ):
            dataframe[column] = dataframe[column].map(
                lambda value: pd.NA
                if isinstance(value, str) and not value.strip()
                else value
            )

    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
    blank_columns = dataframe.columns[dataframe.isna().all()].tolist()
    dataframe = dataframe.drop(columns=blank_columns)
    rows_before_blank_drop = len(dataframe)
    dataframe = dataframe.dropna(how="all")
    blank_rows_removed = rows_before_blank_drop - len(dataframe)

    if dataframe.shape[1] == 0:
        raise ValueError("The uploaded file contains no columns with data.")

    type_conversions = _convert_column_types(dataframe)
    dataframe = dataframe.replace([np.inf, -np.inf], np.nan)

    converted_empty_columns = dataframe.columns[dataframe.isna().all()].tolist()
    dataframe = dataframe.drop(columns=converted_empty_columns)
    blank_columns.extend(converted_empty_columns)
    rows_before_converted_blank_drop = len(dataframe)
    dataframe = dataframe.dropna(how="all")
    blank_rows_removed += rows_before_converted_blank_drop - len(dataframe)

    if dataframe.shape[1] == 0:
        raise ValueError("The uploaded file contains no columns with usable data.")

    duplicate_rows_removed = int(dataframe.duplicated().sum())
    dataframe = dataframe.drop_duplicates().reset_index(drop=True)

    cleaning_report: dict[str, object] = {
        "original_rows": original_rows,
        "original_columns": original_columns,
        "cleaned_rows": int(dataframe.shape[0]),
        "cleaned_columns": int(dataframe.shape[1]),
        "renamed_columns": renamed_columns,
        "dropped_empty_columns": blank_columns,
        "dropped_empty_rows": blank_rows_removed,
        "dropped_duplicate_rows": duplicate_rows_removed,
        "type_conversions": type_conversions,
        "missing_values_imputed": False,
        "outliers_removed": False,
    }
    eda_report = _build_eda_report(dataframe)
    missing_values = eda_report["missing_values"]
    missing_by_column = missing_values["by_column"]
    llm_missing_values = {
        **missing_values,
        "by_column": dict(
            sorted(
                missing_by_column.items(),
                key=lambda item: item[1]["count"],
                reverse=True,
            )[:_MAX_PROFILE_COLUMNS]
        ),
        "additional_columns_omitted": max(
            0, len(missing_by_column) - _MAX_PROFILE_COLUMNS
        ),
    }
    summary = {
        "purpose": "Analyze this cleaned dataset and identify useful trends, anomalies, and potential KPIs.",
        "dataset_profile": eda_report["dataset"],
        "data_quality": {
            "cleaning": cleaning_report,
            "missing_values": llm_missing_values,
        },
        "metrics": {
            "numeric": eda_report["numeric_metrics"],
            "categorical": eda_report["categorical_metrics"],
            "datetime": eda_report["datetime_metrics"],
        },
        "high_correlations": eda_report["high_correlations"],
        "highlights": eda_report["highlights"],
        "interpretation_note": (
            "Numeric aggregates are descriptive and are not business KPIs unless their "
            "meaning and units are confirmed. Correlation does not imply causation. "
            "Category labels and raw records are not included."
        ),
    }
    llm_summary = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)

    return AnalysisResult(
        dataframe=dataframe,
        cleaning_report=cleaning_report,
        eda_report=eda_report,
        llm_summary=llm_summary,
    )
