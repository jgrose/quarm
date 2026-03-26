"""
tools_web.py — Web browsing and search tools for QUARM agents.
browse_url: Headless Chromium via Playwright, returns markdown.
web_search: DuckDuckGo search, returns top results.
"""

import logging
import sqlite3
import os
import time
from datetime import datetime, timezone

log = logging.getLogger("quarm.tools_web")

# ── URL cache (SQLite, 1-hour TTL) ──────────────────────────────────────────

_CACHE_DB = os.path.join(os.path.dirname(__file__), "url_cache.db")
_CACHE_TTL = 3600  # 1 hour


def _init_cache():
    c = sqlite3.connect(_CACHE_DB)
    c.execute("""
        CREATE TABLE IF NOT EXISTS url_cache (
            url TEXT PRIMARY KEY,
            content TEXT,
            cached_at REAL
        )
    """)
    c.commit()
    return c


def _get_cached(url: str) -> str | None:
    try:
        c = sqlite3.connect(_CACHE_DB)
        row = c.execute("SELECT content, cached_at FROM url_cache WHERE url=?", (url,)).fetchone()
        if row and (time.time() - row[1]) < _CACHE_TTL:
            return row[0]
    except Exception:
        pass
    return None


def _set_cached(url: str, content: str):
    try:
        c = sqlite3.connect(_CACHE_DB)
        c.execute(
            "INSERT OR REPLACE INTO url_cache (url, content, cached_at) VALUES (?, ?, ?)",
            (url, content, time.time()),
        )
        c.commit()
    except Exception:
        pass


_init_cache()

# ── Browse URL ───────────────────────────────────────────────────────────────

_browser = None
_playwright = None


def _get_browser():
    """Lazy-init singleton Playwright browser."""
    global _browser, _playwright
    if _browser is None:
        from playwright.sync_api import sync_playwright
        _playwright = sync_playwright().start()
        _browser = _playwright.chromium.launch(headless=True)
        log.info("Playwright Chromium launched")
    return _browser


def browse_url(url: str) -> str:
    """
    Load a web page using headless Chromium and return its content as markdown.
    Handles JavaScript-rendered pages. Results are cached for 1 hour.

    Args:
        url: The URL to browse

    Returns:
        The page content converted to markdown (max 8000 chars)
    """
    # Check cache first
    cached = _get_cached(url)
    if cached:
        log.info(f"Cache hit: {url}")
        return cached

    try:
        browser = _get_browser()
        page = browser.new_page()
        page.set_default_timeout(15000)
        page.goto(url, wait_until="domcontentloaded")
        # Wait a bit for JS to render
        page.wait_for_timeout(2000)
        html = page.content()
        page.close()

        # Convert to markdown
        from markdownify import markdownify
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        # Remove script/style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        md = markdownify(str(soup), heading_style="ATX", strip=["img"])
        # Clean up excessive whitespace
        import re
        md = re.sub(r'\n{3,}', '\n\n', md).strip()
        result = md[:8000]

        _set_cached(url, result)
        return result

    except Exception as e:
        return f"Error browsing {url}: {e}"


# ── Web Search ───────────────────────────────────────────────────────────────

def web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo and return top results.

    Args:
        query: The search query

    Returns:
        Top 5 search results with title, URL, and snippet
    """
    try:
        from duckduckgo_search import DDGS
        results = DDGS().text(query, max_results=5)
        if not results:
            return f"No results found for: {query}"

        output = []
        for i, r in enumerate(results, 1):
            output.append(
                f"{i}. **{r.get('title', 'Untitled')}**\n"
                f"   URL: {r.get('href', '')}\n"
                f"   {r.get('body', '')}"
            )
        return "\n\n".join(output)

    except Exception as e:
        return f"Search error: {e}"
