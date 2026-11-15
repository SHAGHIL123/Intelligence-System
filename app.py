import hashlib

import pandas as pd
import streamlit as st

from dashboard_builder import (
    DashboardValidationError,
    build_dashboard_figures,
    calculate_kpi,
)
from data_loader import analyze_upload
from llm_service import (
    GeminiConfigurationError,
    GeminiRequestError,
    generate_dashboard_plan,
    restore_dashboard_plan,
)

st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #ffffff;
        color: #000000;
    }

    .stApp,
    .stApp [data-testid="stMarkdownContainer"],
    .stApp [data-testid="stMarkdownContainer"] p,
    .stApp [data-testid="stMarkdownContainer"] li,
    .stApp label,
    .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricValue"],
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4 {
        color: #000000;
    }

    .block-container {
        max-width: 1080px;
        padding-top: 3.5rem;
        padding-bottom: 4rem;
    }

    .eyebrow {
        color: #000000;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }

    .hero-title {
        color: #000000;
        font-size: clamp(2.2rem, 5vw, 3.5rem);
        font-weight: 750;
        letter-spacing: -0.045em;
        line-height: 1.08;
        margin: 0;
    }

    .hero-copy {
        color: #000000;
        font-size: 1.08rem;
        line-height: 1.65;
        margin-top: 1rem;
        max-width: 620px;
    }

    .st-key-upload_card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.06);
        margin-top: 2rem;
        padding: 1.5rem 1.75rem 1.75rem;
    }

    .upload-heading {
        color: #000000;
        font-size: 1.15rem;
        font-weight: 650;
        margin-bottom: 0.25rem;
    }

    .upload-caption {
        color: #000000;
        font-size: 0.92rem;
        margin-bottom: 1.25rem;
    }

    [data-testid="stFileUploader"] section {
        background: #ffffff;
        border: 1px dashed #9ca3af;
        border-radius: 14px;
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section * {
        color: #000000 !important;
    }

    [data-testid="stFileUploader"] button {
        background: #ffffff !important;
        border: 1px solid #9ca3af !important;
        color: #000000 !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background: #f3f4f6 !important;
        border-color: #6b7280 !important;
        color: #000000 !important;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: #6b7280;
    }

    .stApp [data-testid="stMarkdownContainer"] p {
        color: #000000;
    }

    .st-key-video_card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.06);
        margin-top: 2rem;
        padding: 1.5rem;
    }

    .video-heading {
        color: #000000;
        font-size: 1.15rem;
        font-weight: 650;
        margin-bottom: 0.25rem;
    }

    .video-caption {
        color: #000000;
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }

    .stButton > button {
        background: linear-gradient(110deg, #2563eb 0%, #3b82f6 52%, #06b6d4 100%);
        border: 1px solid #2563eb;
        border-radius: 10px;
        color: #ffffff;
        font-size: 0.98rem;
        font-weight: 650;
        min-height: 3rem;
        transition: filter 150ms ease, transform 150ms ease;
    }

    .stButton > button:hover {
        background: linear-gradient(110deg, #1d4ed8 0%, #2563eb 52%, #0891b2 100%);
        border-color: #1d4ed8;
        color: #ffffff;
        filter: brightness(1.05);
        transform: translateY(-1px);
    }

    @media (max-width: 640px) {
        .block-container {
            padding: 2rem 1rem;
        }

        .st-key-upload_card {
            padding: 1.25rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

upload_column, video_column = st.columns([1, 1], gap="large", vertical_alignment="center")

with upload_column:
    st.markdown('<div class="eyebrow">Data Insighter</div>', unsafe_allow_html=True)
    st.markdown(
        '<h1 class="hero-title">Make your data<br>tell a story.</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-copy">Upload a dataset to get started. '
        'Bring your data into one clear, focused workspace.</p>',
        unsafe_allow_html=True,
    )

    with st.container(key="upload_card"):
        st.markdown('<div class="upload-heading">Upload your dataset</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="upload-caption">Choose a CSV or Excel file from your device.</div>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Browse files",
            type=["csv", "xlsx"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

with video_column:
    with st.container(key="video_card"):
        st.markdown('<div class="video-heading">A clearer view of your data</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="video-caption">See how data can become a story.</div>',
            unsafe_allow_html=True,
        )
        st.video(
            "https://static.streamlit.io/videos/hero-video.mp4",
            loop=True,
            autoplay=True,
            muted=True,
        )

current_fingerprint = (
    hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if uploaded_file is not None
    else None
)
if st.session_state.get("dashboard_file_fingerprint") != current_fingerprint:
    st.session_state.pop("dashboard_data", None)
    st.session_state.pop("dashboard_plan", None)
    st.session_state.pop("dashboard_file_fingerprint", None)

if analyze_clicked:
    if uploaded_file is None:
        st.error("Choose a CSV or Excel file before analyzing.")
    else:
        try:
            with st.spinner("Cleaning and analyzing your data..."):
                uploaded_file.seek(0)
                analysis = analyze_upload(uploaded_file, uploaded_file.name)
            st.session_state["dashboard_data"] = analysis
            st.session_state.pop("dashboard_plan", None)
            st.session_state["dashboard_file_fingerprint"] = current_fingerprint
        except ValueError as error:
            st.error(str(error))

analysis = st.session_state.get("dashboard_data")
if analysis is not None:
    st.divider()
    st.header("1. Data analysis summary")
    profile = analysis.eda_report["dataset"]
    profile_columns = st.columns(4)
    profile_columns[0].metric("Rows", f"{profile['rows']:,}")
    profile_columns[1].metric("Columns", f"{profile['columns']:,}")
    profile_columns[2].metric(
        "Missing cells",
        f"{analysis.eda_report['missing_values']['missing_cells']:,}",
    )
    profile_columns[3].metric(
        "Duplicate rows removed",
        f"{analysis.cleaning_report['dropped_duplicate_rows']:,}",
    )

    if analysis.eda_report["highlights"]:
        st.subheader("Key points")
        for highlight in analysis.eda_report["highlights"]:
            st.markdown(f"- {highlight}")

    numeric_metrics = analysis.eda_report["numeric_metrics"]
    if numeric_metrics:
        st.subheader("Numeric metrics")
        st.dataframe(
            pd.DataFrame.from_dict(numeric_metrics, orient="index"),
            use_container_width=True,
        )

    categorical_metrics = analysis.eda_report["categorical_metrics"]
    if categorical_metrics:
        st.subheader("Category profile")
        st.dataframe(
            pd.DataFrame.from_dict(categorical_metrics, orient="index"),
            use_container_width=True,
        )

    datetime_metrics = analysis.eda_report["datetime_metrics"]
    if datetime_metrics:
        st.subheader("Date ranges")
        st.dataframe(
            pd.DataFrame.from_dict(datetime_metrics, orient="index"),
            use_container_width=True,
        )

    with st.expander("Data quality and cleaned data"):
        st.json(analysis.cleaning_report)
        st.json(analysis.eda_report["missing_values"])
        st.dataframe(analysis.dataframe.head(100), use_container_width=True)

    st.header("2. AI-designed dashboard")
    if st.button("Generate dashboard and business insights", type="primary"):
        st.session_state.pop("dashboard_plan", None)
        try:
            with st.spinner("Designing a dashboard and deriving business insights..."):
                columns = [
                    {"name": str(column), "dtype": str(dtype)}
                    for column, dtype in analysis.dataframe.dtypes.items()
                ]
                try:
                    api_key = st.secrets.get("GEMINI_API_KEY")
                except FileNotFoundError:
                    api_key = None
                plan = generate_dashboard_plan(
                    analysis.llm_summary,
                    columns,
                    api_key=api_key,
                )
            st.session_state["dashboard_plan"] = plan.model_dump(mode="json")
        except (DashboardValidationError, GeminiConfigurationError, GeminiRequestError) as error:
            st.error(str(error))

    stored_plan = st.session_state.get("dashboard_plan")
    plan = restore_dashboard_plan(stored_plan) if stored_plan is not None else None
    if stored_plan is not None and plan is None:
        st.session_state.pop("dashboard_plan", None)
        st.info("The saved dashboard response is invalid. Generate the dashboard again.")
    elif plan is not None and not isinstance(stored_plan, dict):
        st.session_state["dashboard_plan"] = plan.model_dump(mode="json")

    if plan is not None:
        st.subheader(plan.dashboard_title)
        st.write(plan.dashboard_description)

        if plan.kpis:
            st.subheader("Key indicators")
            metric_columns = st.columns(min(len(plan.kpis), 4))
            for index, kpi in enumerate(plan.kpis):
                try:
                    value = calculate_kpi(
                        analysis.dataframe,
                        kpi.column,
                        kpi.aggregation,
                    )
                except (DashboardValidationError, KeyError) as error:
                    st.warning(f"Skipped KPI '{kpi.label}': {error}")
                    continue
                if isinstance(value, pd.Timestamp):
                    display_value = value.strftime("%b %d, %Y")
                elif isinstance(value, float):
                    display_value = f"{value:,.2f}"
                else:
                    display_value = f"{value:,}"
                metric_columns[index % len(metric_columns)].metric(
                    kpi.label,
                    display_value,
                )

        st.subheader("Dashboard")
        figures, chart_warnings = build_dashboard_figures(
            analysis.dataframe,
            plan,
        )
        for start in range(0, len(figures), plan.layout_columns):
            chart_columns = st.columns(plan.layout_columns)
            for chart_column, (index, (title, figure)) in zip(
                chart_columns,
                enumerate(figures[start : start + plan.layout_columns], start=start),
            ):
                with chart_column:
                    st.plotly_chart(
                        figure,
                        use_container_width=True,
                        key=f"chart_{index}_{title}",
                    )
        for warning in chart_warnings:
            st.warning(warning)
        if not figures and not chart_warnings:
            st.info("Gemini did not recommend any charts for this dataset.")

        st.header("3. Business insights")
        if plan.business_insights:
            for insight in plan.business_insights:
                st.subheader(insight.headline)
                st.write(insight.observation)
                st.markdown(f"**Evidence:** {insight.evidence}")
                st.markdown(
                    f"**Business implication:** {insight.business_implication}"
                )
                st.markdown(f"**Suggested action:** {insight.suggested_action}")
        else:
            st.info(
                "Gemini did not return evidence-backed business insights for this dataset."
            )