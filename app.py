from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


SYSTEM_PROMPT = """You are a practical personal assistant.
Be concise, useful, and honest. For creative tasks, make reasonable assumptions
and produce the requested output instead of asking for more details. If the user
says to continue, proceed from the prior context. Ask a follow-up question only
when the missing detail is truly required. If you are unsure, say so instead of
inventing facts. Refuse unsafe requests briefly and offer a safer alternative
when possible."""


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
    print("OSS model is ready. Starting Streamlit UI.", flush=True)


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
    selected_model = oss_model if backend == "oss" else gemini_model
    llm = build_llm(backend, selected_model, max_tokens, temperature)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *ui_history_to_messages(history, max_turns),
        HumanMessage(content=user_message),
    ]
    return message_to_text(llm.invoke(messages))


def launch_ui(args: argparse.Namespace) -> None:
    download_oss_model(default_model_name("oss", args.model))
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

    if prompt := st.chat_input("Ask the assistant..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking"):
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
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
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
