from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from assistant_service.models import (
    answer_from_history,
    default_model_name,
    download_oss_model,
    estimate_token_count,
)


def launch_ui(args: Any) -> None:
    download_oss_model(default_model_name("oss", args.model))
    print("Starting Streamlit UI.", flush=True)
    command = [
        "streamlit",
        "run",
        str(Path(__file__).resolve().parents[1] / "app.py"),
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
                tokens_per_second_value = output_tokens / max(latency_ms / 1000, 0.001)
                st.markdown(answer)
                st.caption(f"Response time: {latency_ms / 1000:.2f}s")
                st.caption(f"Output speed: {tokens_per_second_value:.1f} tokens/sec")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "latency_ms": latency_ms,
                "output_tokens": output_tokens,
                "tokens_per_second": tokens_per_second_value,
            }
        )
        st.session_state.messages = st.session_state.messages[-max_turns * 2 :]
