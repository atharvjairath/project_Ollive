from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re
from typing import Any
from urllib import request as urlrequest

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field


SYSTEM_PROMPT = """You are a practical personal assistant.
Be concise, useful, and honest. For creative tasks, make reasonable assumptions
and produce the requested output instead of asking for more details. If the user
says to continue, proceed from the prior context. Ask a follow-up question only
when the missing detail is truly required. If you are unsure, say so instead of
inventing facts. Refuse unsafe requests briefly and offer a safer alternative
when possible."""


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    history: list[dict[str, str]] = Field(default_factory=list)
    max_turns: int = Field(default=6, ge=1, le=20)
    max_tokens: int = Field(default=1024, ge=16, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)
    system_prompt: str | None = None


class GenerateResponse(BaseModel):
    text: str
    model: str
    latency_ms: int
    estimated_output_tokens: int
    tokens_per_second: float


app = FastAPI(
    title="Qwen OSS Assistant API",
    description="Public API for Qwen/Qwen2.5-0.5B-Instruct assistant inference.",
    version="0.1.0",
)


@dataclass
class PersonalAssistant:
    """Small LangChain assistant with rolling short-term memory."""

    llm: BaseChatModel
    max_turns: int = 6
    history: list[BaseMessage] = field(default_factory=list)

    def chat(self, user_message: str) -> str:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *self.history,
            HumanMessage(content=user_message),
        ]
        response = self.llm.invoke(messages)
        answer = message_to_text(response)

        self.history.extend([HumanMessage(content=user_message), AIMessage(content=answer)])
        self.history = self.history[-self.max_turns * 2 :]
        return answer

    def reset(self) -> None:
        self.history.clear()

    def transcript(self) -> str:
        if not self.history:
            return "No conversation history yet."
        lines = []
        for message in self.history:
            role = "assistant" if isinstance(message, AIMessage) else "user"
            lines.append(f"{role}: {message_to_text(message)}")
        return "\n".join(lines)


def message_to_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return clean_model_output(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return clean_model_output("\n".join(parts))
    return clean_model_output(str(content))


def clean_model_output(text: str) -> str:
    text = text.strip()
    if "<|im_start|>assistant" in text:
        text = text.rsplit("<|im_start|>assistant", 1)[-1]
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
    text = re.sub(r"^(assistant|system|user)\s*", "", text.strip(), flags=re.IGNORECASE)
    return text.strip()


def estimate_token_count(text: str) -> int:
    """Provider-agnostic fallback for rough output throughput reporting."""
    return max(1, round(len(text) / 4))


def oss_device_index() -> int | None:
    try:
        import torch

        return 0 if torch.cuda.is_available() else None
    except Exception:
        return None


@lru_cache(maxsize=4)
def build_llm(
    backend: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
) -> BaseChatModel:
    if backend == "oss":
        from langchain_huggingface import ChatHuggingFace

        return ChatHuggingFace.from_model_id(
            model_id=model_name,
            task="text-generation",
            backend="pipeline",
            device=oss_device_index(),
            pipeline_kwargs={
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "temperature": temperature,
            },
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking_budget=0,
    )


def default_model_name(backend: str, model: str | None = None) -> str:
    if model:
        return model
    if backend == "oss":
        return os.getenv("OSS_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    return os.getenv("GEMINI_MODEL", "gemini-3.5-flash")


def build_assistant(args: argparse.Namespace) -> PersonalAssistant:
    model_name = default_model_name(args.backend, args.model)
    llm = build_llm(args.backend, model_name, args.max_tokens, args.temperature)
    return PersonalAssistant(llm=llm, max_turns=args.max_turns)


def download_oss_model(model_name: str) -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading/checking OSS model cache: {model_name}", flush=True)
    snapshot_download(repo_id=model_name)
    print("OSS model is ready.", flush=True)


def ui_history_to_messages(history: list[dict[str, Any]], max_turns: int) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for item in history[-max_turns * 2 :]:
        role = item.get("role")
        content = str(item.get("content", ""))
        if role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "user":
            messages.append(HumanMessage(content=content))
    return messages


def answer_from_history(
    user_message: str,
    history: list[dict[str, Any]],
    backend: str,
    oss_model: str,
    gemini_model: str,
    max_turns: int,
    max_tokens: int,
    temperature: float,
) -> str:
    if backend == "oss" and os.getenv("OSS_API_URL"):
        return call_oss_api(
            user_message,
            history,
            max_turns=max_turns,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    selected_model = oss_model if backend == "oss" else gemini_model
    llm = build_llm(backend, selected_model, max_tokens, temperature)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *ui_history_to_messages(history, max_turns),
        HumanMessage(content=user_message),
    ]
    return message_to_text(llm.invoke(messages))


def call_oss_api(
    user_message: str,
    history: list[dict[str, Any]],
    max_turns: int,
    max_tokens: int,
    temperature: float,
) -> str:
    api_url = os.environ["OSS_API_URL"].rstrip("/")
    payload = {
        "prompt": user_message,
        "history": [
            {"role": item.get("role", ""), "content": str(item.get("content", ""))}
            for item in history
            if item.get("role") in {"user", "assistant"}
        ],
        "max_turns": max_turns,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urlrequest.Request(
        f"{api_url}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    return clean_model_output(str(data["text"]))


def generate_oss_response(request: GenerateRequest) -> GenerateResponse:
    model_name = default_model_name("oss")
    llm = build_llm("oss", model_name, request.max_tokens, request.temperature)
    messages = [
        SystemMessage(content=request.system_prompt or SYSTEM_PROMPT),
        *ui_history_to_messages(request.history, request.max_turns),
        HumanMessage(content=request.prompt),
    ]

    start = time.perf_counter()
    answer = message_to_text(llm.invoke(messages))
    latency_ms = round((time.perf_counter() - start) * 1000)
    output_tokens = estimate_token_count(answer)
    tokens_per_second = output_tokens / max(latency_ms / 1000, 0.001)

    return GenerateResponse(
        text=answer,
        model=model_name,
        latency_ms=latency_ms,
        estimated_output_tokens=output_tokens,
        tokens_per_second=round(tokens_per_second, 2),
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


def launch_ui(args: argparse.Namespace) -> None:
    download_oss_model(default_model_name("oss", args.model))
    print("Starting Streamlit UI.", flush=True)
    command = [
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--browser.gatherUsageStats",
        "false",
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none",
        "--server.address",
        args.server_name,
        "--server.port",
        str(args.server_port),
        "--",
        "--streamlit",
    ]
    raise SystemExit(subprocess.call(command))


def render_streamlit_ui() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Assistant Comparison",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Settings")
        backend_label = st.radio(
            "Model",
            ["Gemini", "Open source"],
            horizontal=True,
            index=0,
        )
        backend = "frontier" if backend_label == "Gemini" else "oss"

        gemini_model = st.text_input("Gemini model", value=default_model_name("frontier"))
        oss_model = st.text_input("OSS model", value=default_model_name("oss"))
        api_key = st.text_input("Gemini API key", type="password")
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key

        max_turns = st.slider("Memory turns", 1, 12, 6)
        max_tokens = st.slider("Max tokens", 256, 4096, 1024, step=128)
        temperature = st.slider("Temperature", 0.0, 1.5, 0.7, step=0.1)

        latencies = [
            message["latency_ms"]
            for message in st.session_state.messages
            if message.get("role") == "assistant" and message.get("latency_ms") is not None
        ]
        tokens_per_second = [
            message["tokens_per_second"]
            for message in st.session_state.messages
            if message.get("role") == "assistant"
            and message.get("tokens_per_second") is not None
        ]
        if latencies:
            st.divider()
            st.metric("Last latency", f"{latencies[-1] / 1000:.2f}s")
            st.metric("Average latency", f"{sum(latencies) / len(latencies) / 1000:.2f}s")
        if tokens_per_second:
            st.metric("Last tokens/sec", f"{tokens_per_second[-1]:.1f}")
            st.metric(
                "Average tokens/sec",
                f"{sum(tokens_per_second) / len(tokens_per_second):.1f}",
            )

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    active_name = gemini_model if backend == "frontier" else oss_model
    st.title("Assistant Comparison")
    st.caption("Multi-turn personal assistant using LangChain memory and swappable models.")
    st.info(f"Active model: {backend_label} / {active_name}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("latency_ms") is not None:
                st.caption(f"Response time: {message['latency_ms'] / 1000:.2f}s")
            if message["role"] == "assistant" and message.get("tokens_per_second") is not None:
                st.caption(f"Output speed: {message['tokens_per_second']:.1f} tokens/sec")

    if prompt := st.chat_input("Ask the assistant..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking"):
                start = time.perf_counter()
                try:
                    answer = answer_from_history(
                        prompt,
                        st.session_state.messages[:-1],
                        backend,
                        oss_model,
                        gemini_model,
                        max_turns,
                        max_tokens,
                        temperature,
                    )
                except Exception as exc:
                    answer = f"Error: {exc}"
                latency_ms = round((time.perf_counter() - start) * 1000)
                output_tokens = estimate_token_count(answer)
                tokens_per_second = output_tokens / max(latency_ms / 1000, 0.001)
                st.markdown(answer)
                st.caption(f"Response time: {latency_ms / 1000:.2f}s")
                st.caption(f"Output speed: {tokens_per_second:.1f} tokens/sec")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "latency_ms": latency_ms,
                "output_tokens": output_tokens,
                "tokens_per_second": tokens_per_second,
            }
        )
        st.session_state.messages = st.session_state.messages[-max_turns * 2 :]


def run_cli(assistant: PersonalAssistant, backend_name: str) -> None:
    print(f"Personal assistant running with backend: {backend_name}")
    print("Commands: /reset, /history, /exit")

    while True:
        try:
            user_message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_message:
            continue
        if user_message == "/exit":
            break
        if user_message == "/reset":
            assistant.reset()
            print("Memory cleared.")
            continue
        if user_message == "/history":
            print(assistant.transcript())
            continue

        try:
            answer = assistant.chat(user_message)
        except Exception as exc:
            print(f"Error: {exc}")
            continue
        print(f"\nAssistant: {answer}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple personal assistant.")
    parser.add_argument("--ui", action="store_true", help="Launch the Streamlit web UI.")
    parser.add_argument("--streamlit", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--backend",
        choices=["oss", "frontier"],
        default="frontier",
        help="Use a Hugging Face open-source model or a hosted frontier model.",
    )
    parser.add_argument(
        "--model",
        help="OSS Hugging Face model name. Defaults to OSS_MODEL or Qwen2.5-0.5B-Instruct.",
    )
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    load_dotenv(".env")
    args = parse_args()
    if args.ui:
        launch_ui(args)
        return
    if args.streamlit:
        render_streamlit_ui()
        return
    assistant = build_assistant(args)
    run_cli(assistant, args.backend)


if __name__ == "__main__":
    main()
