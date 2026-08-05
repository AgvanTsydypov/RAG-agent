from dataclasses import dataclass
import re
import pandas as pd
from edgar import Company, set_identity

from app.config import settings

# Columns that describe the row, not a reporting period.
META_COLS = {
    "concept", "label", "standard_concept",
    "level", "abstract", "dimension", "units", "decimals",
}

_PERIOD_RE = re.compile(r"\d{4}")  # a real period column contains a 4-digit year

@dataclass
class FinancialRow:
    ticker: str
    statement: str
    line_item: str
    concept: str
    period: str
    value: float


def _ensure_identity() -> None:
    if settings.edgar_identity:
        set_identity(settings.edgar_identity)


def _to_number(value) -> float | None:
    """Coerce a cell to a float, handling '$', ',', and '(...)' negatives."""
    if isinstance(value, str):
        text = value.replace(",", "").replace("$", "").strip()
        negative = text.startswith("(") and text.endswith(")")
        text = text.strip("()")
        try:
            number = float(text)
        except ValueError:
            return None
        return -number if negative else number
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def _melt_statement(df: pd.DataFrame, ticker: str, statement: str) -> list[FinancialRow]:
    lower = {c: str(c).lower() for c in df.columns}
    period_cols = [c for c in df.columns if lower[c] not in META_COLS and _PERIOD_RE.search(str(c))]
    label_col = next((c for c in df.columns if lower[c] == "label"), None)
    concept_col = next((c for c in df.columns if lower[c] == "concept"), None)

    rows: list[FinancialRow] = []
    for _, row in df.iterrows():
        label = str(row[label_col]).strip() if label_col else str(row.name).strip()
        concept = str(row[concept_col]).strip() if concept_col else ""
        if not label or label.lower() == "nan":
            continue
        for period in period_cols:
            number = _to_number(row[period])
            if number is None:
                continue
            rows.append(
                FinancialRow(ticker, statement, label, concept, str(period), number)
            )
    return rows


def fetch_income_statement(ticker: str) -> list[FinancialRow]:
    _ensure_identity()
    filings = Company(ticker).get_filings(form="10-K")
    if len(filings) == 0:
        return []

    tenk = filings[0].obj()
    statement = tenk.financials.income_statement()
    df = statement.to_dataframe()
    return _melt_statement(df, ticker, "income_statement")