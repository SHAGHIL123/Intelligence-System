"""Gemini analysis service and validated dashboard response schema."""

from __future__ import annotations

import json
import os
from typing import Literal

from dotenv import load_dotenv
from google.genai.errors import APIError
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError


Aggregation = Literal["sum", "mean", "median", "min", "max", "count", "unique"]
ChartType = Literal["bar", "line", "scatter", "histogram", "pie"]


class KpiSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=60)
    column: str
    aggregation: Aggregation


class ChartSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chart_type: ChartType
    title: str = Field(min_length=1, max_length=100)
    x_column: str
    y_column: str | None = None
    aggregation: Literal["sum", "mean", "median", "count"] | None = None


class BusinessInsight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headline: str = Field(min_length=1, max_length=100)
    observation: str = Field(min_length=1, max_length=500)
    evidence: str = Field(min_length=1, max_length=300)
    business_implication: str = Field(min_length=1, max_length=400)
    suggested_action: str = Field(min_length=1, max_length=300)


class DashboardPlan(BaseModel):
    """A complete dashboard design and evidence-based business insights."""

    model_config = ConfigDict(extra="forbid")

    dashboard_title: str = Field(min_length=1, max_length=100)
    dashboard_description: str = Field(min_length=1, max_length=300)
    layout_columns: Literal[1, 2, 3] = 2
    kpis: list[KpiSpec] = Field(default_factory=list, max_length=6)
    charts: list[ChartSpec] = Field(default_factory=list, max_length=6)
    business_insights: list[BusinessInsight] = Field(default_factory=list, max_length=8)


class GeminiConfigurationError(RuntimeError):
    """Gemini credentials or model configuration are missing."""


class GeminiRequestError(RuntimeError):
    """Gemini could not provide a valid structured analysis."""


def generate_dashboard_plan(
    llm_summary: str,
    columns: list[dict[str, str]],
    *,
    api_key: str | None = None,
    model_name: str | None = None,
) -> DashboardPlan:
    """Request a complete dashboard design and grounded business insights."""
    load_dotenv()
    resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not resolved_key:
        raise GeminiConfigurationError(
            "Gemini API key is missing. Set GEMINI_API_KEY in .env or Streamlit secrets."
        )

    resolved_model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    client = ChatGoogleGenerativeAI(
        model=resolved_model,
        api_key=resolved_key,
        temperature=0,
    ).with_structured_output(DashboardPlan, method="json_schema")

    system_prompt = (
        "You are a careful business data analyst and dashboard designer. Use only "
        "the aggregate dataset summary and available column metadata supplied by "
        "the user. Do not invent values, business definitions, trends, segment "
        "names, or causal explanations. Treat the dashboard fields as an internal "
        "implementation design: provide a useful title and description, a layout "
        "of 1-3 columns, up to 6 meaningful KPIs, and up to 6 chart specifications "
        "chosen to tell a coherent story. Select chart columns only from supplied "
        "metadata. Use histogram for one numeric column; scatter for two numeric "
        "columns; line for date/numeric x and numeric y; bar for category/date/"
        "numeric x (y may be omitted for category counts); pie for categorical x "
        "(with optional numeric y). For bar and pie charts with a numeric y, "
        "select an aggregation (sum, mean, median, or count) appropriate to the "
        "metric and business question. Then provide separate business_insights: "
        "each must state a specific observation, cite evidence from the supplied "
        "aggregate summary, explain the business implication cautiously, and give "
        "a practical suggested action. Do not merely repeat data-quality facts as "
        "insights unless they have an explicit business implication. Never return "
        "code or instructions to execute code."
    )
    human_prompt = (
        "Aggregate analysis summary (JSON):\n"
        f"{llm_summary}\n\n"
        "Available columns (JSON, use exact names):\n"
        f"{json.dumps(columns, ensure_ascii=False)}"
    )

    try:
        response = client.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )
        return (
            response
            if isinstance(response, DashboardPlan)
            else DashboardPlan.model_validate(response)
        )
    except (APIError, OutputParserException, ValidationError) as error:
        raise GeminiRequestError(
            "Gemini did not return a valid dashboard analysis. Please try again."
        ) from error


def restore_dashboard_plan(value: object) -> DashboardPlan | None:
    """Restore a session plan, upgrading plans cached from the previous schema."""
    if isinstance(value, DashboardPlan):
        return value

    if isinstance(value, dict):
        data = value
    else:
        dump = getattr(value, "model_dump", None)
        if not callable(dump):
            return None
        data = dump()

    if not isinstance(data, dict):
        return None

    try:
        return DashboardPlan.model_validate(data)
    except ValidationError:
        if not {"kpis", "charts"}.issubset(data):
            return None
        legacy_data = {
            "dashboard_title": data.get("dashboard_title", "Data dashboard"),
            "dashboard_description": data.get(
                "dashboard_description",
                "Restored from this session. Regenerate the dashboard for updated business insights.",
            ),
            "layout_columns": data.get("layout_columns", 2),
            "kpis": data.get("kpis", []),
            "charts": data.get("charts", []),
            "business_insights": data.get("business_insights", []),
        }
        try:
            return DashboardPlan.model_validate(legacy_data)
        except ValidationError:
            return None
