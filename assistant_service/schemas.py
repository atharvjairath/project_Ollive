from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    history: list[dict[str, str]] = Field(default_factory=list)
    max_turns: int = Field(default=6, ge=1, le=20)
    max_tokens: int = Field(default=1024, ge=16, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)
    system_prompt: str | None = None
    enable_web_search: bool = False


class GenerateResponse(BaseModel):
    text: str
    model: str
    latency_ms: int
    estimated_output_tokens: int
    tokens_per_second: float
    guardrail_action: str = "allowed"
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    trace_id: str | None = None


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    action: str
    message: str = ""
