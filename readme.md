# Data Analysis Dashboard

A Streamlit application for turning CSV and Excel files into an interactive dashboard and an AI-generated analysis.

## How it works

1. Upload a `.csv` or `.xlsx` file and select **Analyze**.
2. Clean and profile the data with Pandas and NumPy; review the local summary.
3. Select **Generate dashboard and business insights** to send aggregate metrics and column metadata to Gemini through LangChain.
4. Review the rendered dashboard, then read the separate evidence-based business insights below it.

## Technology

- **Streamlit** for the application interface
- **Pandas and NumPy** for data analysis
- **Plotly** for interactive visualizations
- **LangChain and Google Generative AI** for AI-assisted analysis
- **OpenPyXL** for Excel file support

## Data loading and analysis

`data_loader.py` provides `analyze_upload(uploaded_file, filename=None)`. It accepts a file-like upload and returns an `AnalysisResult` containing:

- `dataframe`: cleaned Pandas DataFrame
- `cleaning_report`: renamed columns, removed empty/duplicate data, and inferred type conversions
- `eda_report`: missing-value profile, numeric metrics, categorical/date profiles, outlier counts, correlations, and highlights
- `llm_summary`: compact JSON with aggregate analysis context; it does not include raw rows or category labels

Partial missing values are preserved, and outliers are reported rather than removed. Example:

```python
from data_loader import analyze_upload

result = analyze_upload(uploaded_file)
print(result.llm_summary)
```

## Gemini dashboard flow

The **Analyze** button runs local cleaning and EDA first. It displays the dataset profile, numeric and categorical metrics, date ranges, highlights, and data-quality details. After reviewing these results, select **Generate dashboard and business insights** to send `llm_summary` and the available column names/types to Gemini. The request runs during the Streamlit interaction and shows a loading indicator; it is not a separate background worker. Dataset rows are kept in the app for chart rendering and are not included in the Gemini prompt.

Gemini returns an internal dashboard design (title, layout, KPIs, and chart specifications) separately from evidence-based business insights. The UI renders the dashboard first and displays business observations, supporting evidence, implications, and suggested actions below it; it does not display the model's internal analysis/design prompt as a user-facing summary. The app calculates KPIs locally and builds charts using Plotly—model-generated code is never executed. Invalid chart suggestions are skipped with a warning so other charts can still render.

Set `GEMINI_API_KEY` in `.env` (or configure it in Streamlit secrets when deployed). Optionally set `GEMINI_MODEL` to select a Gemini model; the default is `gemini-2.5-flash`.

## Run locally

Use Python with `pip` installed. From the project directory, create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file in the project directory:

```text
GEMINI_API_KEY=your-api-key
```

Keep API keys private and do not commit `.env` to source control.

Start the Streamlit app:

```powershell
streamlit run app.py
```

## Current status

The Streamlit upload interface, CSV/XLSX analysis, Gemini integration, and validated Plotly dashboard rendering are connected. Gemini requests require a valid API key and network access.
