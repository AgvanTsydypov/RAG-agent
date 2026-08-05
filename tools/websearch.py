from app.config import settings

try:
    from tavily import TavilyClient
except ImportError:  # keep the import optional until the package is installed
    TavilyClient = None

_client = (
    TavilyClient(api_key=settings.tavily_api_key)
    if TavilyClient and settings.tavily_api_key
    else None
)


def web_search(query: str, max_results: int = 3) -> str:
    """Search the live web for recent or external facts NOT in the SEC filings
    (current events, stock prices, news). Returns ranked snippets with URLs.
    """
    if _client is None:
        return "Web search is not configured (missing TAVILY_API_KEY)."
    response = _client.search(query, max_results=max_results)
    results = response.get("results", [])
    if not results:
        return "No web results found."
    return "\n\n".join(
        f"[{i}] {r.get('title')} ({r.get('url')})\n{r.get('content')}"
        for i, r in enumerate(results, start=1)
    )