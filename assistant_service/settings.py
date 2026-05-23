from __future__ import annotations

import os


SYSTEM_PROMPT = """You are a practical personal assistant.
Be concise, useful, and honest. For creative tasks, make reasonable assumptions
and produce the requested output instead of asking for more details. If the user
says to continue, proceed from the prior context. Ask a follow-up question only
when the missing detail is truly required. If you are unsure, say so instead of
inventing facts. Refuse unsafe requests briefly and offer a safer alternative
when possible."""


def default_model_name(backend: str, model: str | None = None) -> str:
    if model:
        return model
    if backend == "oss":
        return os.getenv("OSS_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    return os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
