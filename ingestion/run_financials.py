from ingestion.fetch_financials import fetch_income_statement
from ingestion.store import delete_financials, ensure_schema, insert_financials

TICKERS = ["AAPL", "MSFT"]


def main() -> None:
    ensure_schema()  # adds the financials table if missing
    for ticker in TICKERS:
        rows = fetch_income_statement(ticker)
        if not rows:
            print(f"[{ticker}] no income statement rows, skipping")
            continue
        delete_financials(ticker)
        insert_financials(
            [
                (r.ticker, r.statement, r.line_item, r.concept, r.period, r.value)
                for r in rows
            ]
        )
        print(f"[{ticker}] stored {len(rows)} financial rows")


if __name__ == "__main__":
    main()