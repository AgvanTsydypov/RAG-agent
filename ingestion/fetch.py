from dataclasses import dataclass

from edgar import Company, set_identity

from app.config import settings

# 10-K narrative sections we want in the vector store, mapped to the
# attribute names exposed by edgartools' TenK object.
SECTION_ATTRS = {
    "business_description": "business_description",
    "risk_factors": "risk_factors",
    "mda": "mda",  # Management's Discussion & Analysis
}


@dataclass
class SectionText:
    ticker: str
    form_type: str
    filing_date: str  # ISO date; psycopg casts it into the DATE column
    section: str
    text: str


def _ensure_identity() -> None:
    # The SEC blocks requests without a proper User-Agent identity.
    if settings.edgar_identity:
        set_identity(settings.edgar_identity)


def fetch_tenk_sections(ticker: str) -> list[SectionText]:
    """Fetch the latest 10-K for a ticker and return its narrative sections."""
    _ensure_identity()

    filings = Company(ticker).get_filings(form="10-K")
    if len(filings) == 0:
        return []

    filing = filings[0]  # index 0 is the most recent filing
    tenk = filing.obj()  # typed TenK object with named sections + financials

    results: list[SectionText] = []
    for section_name, attr in SECTION_ATTRS.items():
        text = getattr(tenk, attr, None)
        if text and text.strip():
            results.append(
                SectionText(
                    ticker=ticker,
                    form_type="10-K",
                    filing_date=str(filing.filing_date),
                    section=section_name,
                    text=text,
                )
            )
    return results