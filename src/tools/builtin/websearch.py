"""Web search built-in tool using DuckDuckGo (no API key required)."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DDGS_AVAILABLE = False
try:
    from ddgs import DDGS as _DDGS

    _DDGS_AVAILABLE = True
except ImportError:
    logger.warning(
        "ddgs is not installed. "
        "The web_search tool will return an error. "
        "Install with: pip install ddgs"
    )


def web_search(query: str, max_results: int = 6) -> list[dict] | dict:
    """Search the web using DuckDuckGo and return a list of results."""
    if not _DDGS_AVAILABLE:
        return {
            "error": (
                "ddgs is not installed on the server. "
                "Install it with: pip install ddgs"
            )
        }
    max_results = max(1, min(max_results, 20))
    try:
        with _DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        logger.exception("web_search error")
        return {"error": f"Search failed: {exc}"}
    # Normalise field names to match OpenAI convention (title, url, snippet)
    normalized = [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        }
        for r in results
    ]
    return normalized


# ---------------------------------------------------------------------------
# OpenAI tool schema
# ---------------------------------------------------------------------------

WEB_SEARCH_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Searches the web using DuckDuckGo and returns a list of relevant results "
            "(title, URL, and snippet). Use this to find current information, news, "
            "documentation, or any topic the model may not know about."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1–20, default 6).",
                    "default": 6,
                },
            },
            "required": ["query"],
        },
    },
}
