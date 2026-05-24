from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool


def should_use_web_search(prompt: str) -> bool:
    return bool(
        re.search(
            r"\b(search|web|latest|current|today|news|source|sources|pricing|price)\b",
            prompt,
            re.IGNORECASE,
        )
    )


def search_web(query: str, max_results: int = 3) -> list[dict[str, str]]:
    from ddgs import DDGS

    results = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": str(item.get("title", "")),
                    "url": str(item.get("href", "")),
                    "snippet": str(item.get("body", ""))[:300],
                }
            )
    return results


def search_results_to_context(results: list[dict[str, str]]) -> str:
    lines = ["Use these web search results when relevant. Cite URLs in the answer."]
    for index, result in enumerate(results, start=1):
        lines.append(
            f"{index}. {result['title']}\nURL: {result['url']}\nSnippet: {result['snippet']}"
        )
    return "\n\n".join(lines)


@tool("web_search")
def web_search_tool(query: str) -> str:
    """Search the web for current or external information and return sourced snippets."""

    return json.dumps(search_web(query), indent=2)


WEB_SEARCH_TOOLS = [web_search_tool]


def tool_calls_from_message(message: Any) -> list[dict[str, Any]]:
    return list(getattr(message, "tool_calls", None) or [])


def run_web_search_tool_call(tool_call: dict[str, Any], fallback_query: str) -> tuple[ToolMessage, dict[str, Any]]:
    args = tool_call.get("args") or {}
    query = str(args.get("query") or fallback_query)
    try:
        results = search_web(query)
        error = ""
    except Exception as exc:
        results = []
        error = str(exc)

    metadata = {
        "name": "web_search",
        "query": query,
        "result_count": len(results),
        "results": results,
        "error": error,
    }
    content = search_results_to_context(results) if results else f"Web search returned no results. {error}".strip()
    return (
        ToolMessage(
            content=content,
            tool_call_id=str(tool_call.get("id") or "web_search_call"),
            name="web_search",
        ),
        metadata,
    )
