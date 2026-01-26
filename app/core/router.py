from dataclasses import dataclass
from typing import Literal, Tuple, Optional

from langchain_openai import ChatOpenAI

from app.core.config import OPENAI_API_KEY, LLM_MODEL


# ---------- Fast rule-based router (cheap & deterministic) ----------
def fast_rule_route(user_query: str) -> Tuple[Optional[str], Optional[str]]:
    ql = user_query.lower()

    if any(word in ql for word in ["latest", "today", "current", "news", "2025", "2026", "price", "stock"]):
        return "WEB", "Needs up-to-date information."

    if any(word in ql for word in ["compare", "versus", "vs", "validate", "contrast"]):
        return "BOTH", "Comparison requires docs + web."

    return None, None


# ---------- Types ----------
Route = Literal["DOCS", "WEB", "BOTH"]


@dataclass
class RouteDecision:
    route: Route
    reason: str


# ---------- LLM router (fallback when rules don’t trigger) ----------
def decide_route(user_query: str) -> RouteDecision:
    """
    Decide where to answer from:
    - DOCS: from uploaded documents only
    - WEB: from web search only
    - BOTH: combine documents + web
    """

    # 1) Try fast rule-based routing first
    r, reason = fast_rule_route(user_query)
    if r:
        return RouteDecision(route=r, reason=reason)

    # 2) Fall back to LLM-based router
    router_llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
    )

    prompt = f"""
You are a routing controller for a research assistant.
Choose exactly one route: DOCS, WEB, BOTH.

Rules:
- DOCS: if the question can be answered from the user's uploaded documents.
- WEB: if it requires up-to-date info (latest news, current prices, recent events) or general knowledge not in documents.
- BOTH: if it asks to compare/validate/extend document content with current info from the web.

Return ONLY in this strict format (2 lines):
ROUTE: <DOCS|WEB|BOTH>
REASON: <one short sentence>

User question:
{user_query}
"""

    text = router_llm.invoke(prompt).content.strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    route: Route = "DOCS"
    reason = "Defaulted to DOCS."

    for line in lines:
        if line.upper().startswith("ROUTE:"):
            route = line.split(":", 1)[1].strip().upper()  # type: ignore
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    if route not in {"DOCS", "WEB", "BOTH"}:
        route = "DOCS"

    return RouteDecision(route=route, reason=reason)
