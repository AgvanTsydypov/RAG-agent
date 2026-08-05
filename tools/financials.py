from ingestion.store import query_financials_rows


def query_financials(ticker: str, metric: str) -> str:
    """Look up reported figures from a company's 10-K income statement.

    Use for QUANTITATIVE questions (revenue, net income, operating income...).
    `ticker` is a symbol like "AAPL". `metric` is a keyword matched against
    line-item names. Returns each matching line item per reporting period.
    """
    rows = query_financials_rows(ticker.upper(), metric)
    if not rows:
        return f"No financial data for {ticker.upper()} matching '{metric}'."
    return "\n".join(
        f"{line_item} | {period}: {value}" for line_item, period, value in rows
    )