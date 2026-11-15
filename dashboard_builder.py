"""Validate model-generated dashboard specs and build native Plotly figures."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from llm_service import ChartSpec, DashboardPlan


class DashboardValidationError(ValueError):
    """A dashboard plan refers to unavailable or incompatible data."""


def validate_dashboard_plan(plan: DashboardPlan, dataframe: pd.DataFrame) -> None:
    """Ensure every requested KPI and chart uses available, compatible columns."""
    available = set(dataframe.columns)

    for kpi in plan.kpis:
        if kpi.column not in available:
            raise DashboardValidationError(
                f"The suggested KPI column '{kpi.column}' is not in the dataset."
            )
        values = dataframe[kpi.column]
        is_datetime = is_datetime64_any_dtype(values)
        if kpi.aggregation in ("min", "max") and is_datetime:
            continue
        if kpi.aggregation not in ("count", "unique") and not is_numeric_dtype(values):
            raise DashboardValidationError(
                f"KPI '{kpi.label}' requires a numeric column for "
                f"{kpi.aggregation}, or a date/time column for min or max."
            )

    for chart in plan.charts:
        if chart.x_column not in available:
            raise DashboardValidationError(
                f"The suggested chart column '{chart.x_column}' is not in the dataset."
            )
        if chart.y_column is not None and chart.y_column not in available:
            raise DashboardValidationError(
                f"The suggested chart column '{chart.y_column}' is not in the dataset."
            )

        _validate_chart(chart, dataframe)


def _validate_chart(chart: ChartSpec, dataframe: pd.DataFrame) -> None:
    available = set(dataframe.columns)
    if chart.x_column not in available:
        raise DashboardValidationError(
            f"The suggested chart column '{chart.x_column}' is not in the dataset."
        )
    if chart.y_column is not None and chart.y_column not in available:
        raise DashboardValidationError(
            f"The suggested chart column '{chart.y_column}' is not in the dataset."
        )

    x_values = dataframe[chart.x_column]
    x_is_numeric = is_numeric_dtype(x_values)
    x_is_datetime = is_datetime64_any_dtype(x_values)
    y_is_numeric = (
        chart.y_column is not None
        and is_numeric_dtype(dataframe[chart.y_column])
    )
    if chart.aggregation is not None and chart.chart_type not in ("bar", "pie"):
        raise DashboardValidationError(
            "Aggregation is only supported for bar and pie charts."
        )

    if chart.chart_type == "histogram":
        if chart.y_column is not None:
            raise DashboardValidationError("Histogram charts use only an x column.")
        if not x_is_numeric:
            raise DashboardValidationError("Histogram charts require a numeric x column.")
    elif chart.chart_type in ("line", "scatter"):
        if chart.y_column is None or not y_is_numeric:
            raise DashboardValidationError(
                f"{chart.chart_type.title()} charts require a numeric y column."
            )
        if chart.chart_type == "scatter" and not x_is_numeric:
            raise DashboardValidationError("Scatter charts require a numeric x column.")
        if chart.chart_type == "line" and not (x_is_numeric or x_is_datetime):
            raise DashboardValidationError(
                "Line charts require a numeric or date/time x column."
            )
    elif chart.chart_type == "pie":
        if x_is_numeric or x_is_datetime:
            raise DashboardValidationError("Pie charts require a categorical x column.")
        if chart.y_column is not None and not y_is_numeric:
            raise DashboardValidationError("Pie chart values must use a numeric y column.")
    elif chart.chart_type == "bar" and chart.y_column is not None and not y_is_numeric:
        raise DashboardValidationError(
            "Bar chart y columns must be numeric; omit y to chart category counts."
        )


def calculate_kpi(
    dataframe: pd.DataFrame,
    column: str,
    aggregation: str,
) -> int | float | pd.Timestamp:
    """Calculate a validated KPI from the cleaned dataset."""
    values = dataframe[column]
    if aggregation == "count":
        return int(values.count())
    if aggregation == "unique":
        return int(values.nunique())
    if is_datetime64_any_dtype(values) and aggregation in ("min", "max"):
        result = values.min() if aggregation == "min" else values.max()
        if pd.isna(result):
            raise DashboardValidationError(
                f"KPI column '{column}' has no valid dates to calculate {aggregation}."
            )
        return result

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if aggregation == "sum":
        result = numeric.sum()
    elif aggregation == "mean":
        result = numeric.mean()
    elif aggregation == "median":
        result = numeric.median()
    elif aggregation == "min":
        result = numeric.min()
    elif aggregation == "max":
        result = numeric.max()
    else:
        raise DashboardValidationError(f"Unsupported KPI aggregation: {aggregation}")

    return int(result) if float(result).is_integer() else float(result)


def _build_chart(
    dataframe: pd.DataFrame,
    chart_type: str,
    title: str,
    x_column: str,
    y_column: str | None,
    aggregation: str | None,
) -> go.Figure:
    if chart_type == "histogram":
        return px.histogram(dataframe, x=x_column, title=title)

    if chart_type == "scatter":
        chart_data = dataframe[[x_column, y_column]].dropna()
        if len(chart_data) > 5000:
            chart_data = chart_data.sample(n=5000, random_state=0)
        return px.scatter(chart_data, x=x_column, y=y_column, title=title)

    if chart_type == "line":
        chart_data = dataframe[[x_column, y_column]].dropna().sort_values(x_column)
        if len(chart_data) > 5000:
            chart_data = chart_data.iloc[:: max(1, len(chart_data) // 5000)]
        return px.line(chart_data, x=x_column, y=y_column, title=title)

    if chart_type == "pie":
        if y_column is None:
            chart_data = dataframe[x_column].value_counts().head(15).reset_index()
            chart_data.columns = [x_column, "count"]
            return px.pie(chart_data, names=x_column, values="count", title=title)
        chart_data = (
            dataframe[[x_column, y_column]]
            .dropna()
        )
        grouped = chart_data.groupby(x_column)[y_column]
        if aggregation == "mean":
            chart_data = grouped.mean().reset_index()
        elif aggregation == "median":
            chart_data = grouped.median().reset_index()
        elif aggregation == "count":
            chart_data = grouped.count().reset_index()
        else:
            chart_data = grouped.sum().reset_index()
        chart_data = chart_data.nlargest(15, y_column)
        return px.pie(chart_data, names=x_column, values=y_column, title=title)

    if y_column is None:
        chart_data = dataframe[x_column].value_counts().head(20).reset_index()
        chart_data.columns = [x_column, "count"]
        return px.bar(chart_data, x=x_column, y="count", title=title)

    grouped = dataframe[[x_column, y_column]].dropna().groupby(x_column)[y_column]
    if aggregation == "sum":
        chart_data = grouped.sum().reset_index()
    elif aggregation == "median":
        chart_data = grouped.median().reset_index()
    elif aggregation == "count":
        chart_data = grouped.count().reset_index()
    else:
        chart_data = grouped.mean().reset_index()
    chart_data = chart_data.head(100)
    return px.bar(chart_data, x=x_column, y=y_column, title=title)


def build_dashboard_figures(
    dataframe: pd.DataFrame,
    plan: DashboardPlan,
) -> tuple[list[tuple[str, go.Figure]], list[str]]:
    """Build valid suggested figures and report invalid chart suggestions."""
    figures: list[tuple[str, go.Figure]] = []
    warnings: list[str] = []

    for chart in plan.charts:
        try:
            _validate_chart(chart, dataframe)
            figure = _build_chart(
                dataframe,
                chart.chart_type,
                chart.title,
                chart.x_column,
                chart.y_column,
                chart.aggregation,
            )
        except (DashboardValidationError, KeyError, TypeError, ValueError) as error:
            warnings.append(f"Skipped '{chart.title}': {error}")
            continue
        figures.append((chart.title, figure))

    return figures, warnings
