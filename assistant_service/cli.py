from __future__ import annotations

import argparse

from dotenv import load_dotenv

from assistant_service.models import PersonalAssistant, build_assistant
from assistant_service.ui import launch_ui, render_streamlit_ui


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
