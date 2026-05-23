from __future__ import annotations

import re


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
