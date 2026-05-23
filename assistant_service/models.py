from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any
from urllib import request as urlrequest

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from assistant_service.settings import SYSTEM_PROMPT, default_model_name


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
        answer = message_to_text(self.llm.invoke(messages))
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


def build_assistant(args: Any) -> PersonalAssistant:
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
