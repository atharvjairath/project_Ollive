from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from langchain_core.messages import HumanMessage, SystemMessage

from assistant_service.models import (
    build_llm,
    default_model_name,
    download_oss_model,
    estimate_token_count,
    message_to_text,
    ui_history_to_messages,
)
from assistant_service.observability import logger, tracer
from assistant_service.safety import run_input_guardrails, run_output_guardrails
from assistant_service.schemas import GenerateRequest, GenerateResponse
from assistant_service.settings import SYSTEM_PROMPT
from assistant_service.tools import search_results_to_context, search_web, should_use_web_search


app = FastAPI(
    title="Qwen OSS Assistant API",
    description="Public API for Qwen/Qwen2.5-0.5B-Instruct assistant inference.",
    version="0.1.0",
)


def build_tool_source_footer(tool_calls: list[dict[str, Any]]) -> str:
    source_lines = []
    for tool_call in tool_calls:
        if tool_call["name"] == "web_search":
            for result in tool_call.get("results", []):
                if result.get("title") and result.get("url"):
                    source_lines.append(f"- {result['title']}: {result['url']}")
    return "\n".join(source_lines)


def generate_oss_response(request: GenerateRequest) -> GenerateResponse:
    trace_id = str(uuid.uuid4())
    model_name = default_model_name("oss")
    start = time.perf_counter()
    tool_calls: list[dict[str, Any]] = []

    with tracer.start_as_current_span("generate") as span:
        span.set_attribute("trace_id", trace_id)
        span.set_attribute("model", model_name)
        span.set_attribute("max_tokens", request.max_tokens)
        span.set_attribute("temperature", request.temperature)
        span.set_attribute("web_search.enabled", request.enable_web_search)

        guardrail = run_input_guardrails(request.prompt)
        span.set_attribute("guardrail.input_action", guardrail.action)
        if not guardrail.allowed:
            latency_ms = max(1, round((time.perf_counter() - start) * 1000))
            output_tokens = estimate_token_count(guardrail.message)
            logger.info(
                "observability_event=%s",
                json.dumps(
                    {
                        "trace_id": trace_id,
                        "event": "guardrail_block",
                        "action": guardrail.action,
                        "latency_ms": latency_ms,
                    }
                ),
            )
            return GenerateResponse(
                text=guardrail.message,
                model=model_name,
                latency_ms=latency_ms,
                estimated_output_tokens=output_tokens,
                tokens_per_second=0.0,
                guardrail_action=guardrail.action,
                tool_calls=tool_calls,
                trace_id=trace_id,
            )

        messages = [
            SystemMessage(content=request.system_prompt or SYSTEM_PROMPT),
            *ui_history_to_messages(request.history, request.max_turns),
        ]

        if request.enable_web_search and should_use_web_search(request.prompt):
            with tracer.start_as_current_span("tool.web_search") as tool_span:
                try:
                    search_results = search_web(request.prompt)
                    tool_error = ""
                except Exception as exc:
                    search_results = []
                    tool_error = str(exc)
                tool_calls.append(
                    {
                        "name": "web_search",
                        "query": request.prompt,
                        "result_count": len(search_results),
                        "results": search_results,
                        "error": tool_error,
                    }
                )
                tool_span.set_attribute("tool.name", "web_search")
                tool_span.set_attribute("tool.result_count", len(search_results))
                if tool_error:
                    tool_span.set_attribute("tool.error", tool_error)
                if search_results:
                    messages.append(SystemMessage(content=search_results_to_context(search_results)))

        messages.append(HumanMessage(content=request.prompt))

        with tracer.start_as_current_span("llm.invoke") as llm_span:
            llm = build_llm("oss", model_name, request.max_tokens, request.temperature)
            answer = message_to_text(llm.invoke(messages))
            llm_span.set_attribute("output.estimated_tokens", estimate_token_count(answer))

        source_footer = build_tool_source_footer(tool_calls)
        if source_footer:
            answer = f"{answer}\n\nSources:\n{source_footer}"

        output_guardrail = run_output_guardrails(answer)
        span.set_attribute("guardrail.output_action", output_guardrail.action)
        if not output_guardrail.allowed:
            answer = output_guardrail.message

        latency_ms = max(1, round((time.perf_counter() - start) * 1000))
        output_tokens = estimate_token_count(answer)
        tokens_per_second = output_tokens / max(latency_ms / 1000, 0.001)
        guardrail_action = (
            output_guardrail.action if not output_guardrail.allowed else guardrail.action
        )

        logger.info(
            "observability_event=%s",
            json.dumps(
                {
                    "trace_id": trace_id,
                    "event": "generation",
                    "model": model_name,
                    "latency_ms": latency_ms,
                    "estimated_output_tokens": output_tokens,
                    "tokens_per_second": round(tokens_per_second, 2),
                    "guardrail_action": guardrail_action,
                    "tool_calls": [tool_call["name"] for tool_call in tool_calls],
                }
            ),
        )

        return GenerateResponse(
            text=answer,
            model=model_name,
            latency_ms=latency_ms,
            estimated_output_tokens=output_tokens,
            tokens_per_second=round(tokens_per_second, 2),
            guardrail_action=guardrail_action,
            tool_calls=tool_calls,
            trace_id=trace_id,
        )


@app.on_event("startup")
def load_model_on_startup() -> None:
    model_name = default_model_name("oss")
    download_oss_model(model_name)
    build_llm("oss", model_name, 1024, 0.7)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Qwen OSS Assistant API",
        "model": default_model_name("oss"),
        "docs": "/docs",
        "generate": "POST /generate",
        "observability": "OpenTelemetry spans + structured JSON logs",
        "guardrails": "input and output safety rails",
        "tools": "optional web_search via enable_web_search=true",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": default_model_name("oss")}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    try:
        return await run_in_threadpool(generate_oss_response, request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
