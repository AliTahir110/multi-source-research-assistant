from tavily import TavilyClient

from app.core.config import TAVILY_API_KEY


def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Returns a list of results:
    [{title, url, content}]
    Content is trimmed to reduce prompt noise/cost.
    """
    client = TavilyClient(api_key=TAVILY_API_KEY)
    resp = client.search(
        query=query,
        max_results=max_results,
        include_answer=False,
        include_raw_content=False,
    )

    results = resp.get("results", []) or []

    cleaned = []
    for r in results:
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        content = (r.get("content") or "").strip()

        # Trim content to keep prompt small and focused
        if len(content) > 700:
            content = content[:700] + "..."

        if title and url and content:
            cleaned.append({"title": title, "url": url, "content": content})

    return cleaned
