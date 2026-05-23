from __future__ import annotations

import re

from assistant_service.schemas import GuardrailResult


INPUT_RAILS = [
    (
        "malware",
        re.compile(
            r"(?=.*\b(ransomware|malware|keylogger|credential stealer|reverse shell|botnet)\b)"
            r"(?=.*\b(make|build|write|code|create|deploy|instructions?|steps?|explain how)\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "credential_theft",
        re.compile(
            r"\b(phishing|steal passwords?|bypass login|credential stuffing|session hijack)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "weapons",
        re.compile(
            r"\b(make|build|instructions?|recipe)\b.*\b(explosive|bomb|poison|weapon)\b",
            re.IGNORECASE,
        ),
    ),
]

OUTPUT_RAILS = [
    (
        "malware_output",
        re.compile(
            r"\b(encrypt files|persistence|exfiltrate|payload|keylogger|ransom note)\b",
            re.IGNORECASE,
        ),
    )
]


def run_input_guardrails(prompt: str) -> GuardrailResult:
    for category, pattern in INPUT_RAILS:
        if pattern.search(prompt):
            return GuardrailResult(
                allowed=False,
                action=f"blocked_input:{category}",
                message=(
                    "I can't help with instructions that enable cyber abuse or physical harm. "
                    "I can help with defensive guidance, detection, prevention, or incident response."
                ),
            )
    return GuardrailResult(allowed=True, action="allowed")


def run_output_guardrails(text: str) -> GuardrailResult:
    for category, pattern in OUTPUT_RAILS:
        if pattern.search(text):
            return GuardrailResult(
                allowed=False,
                action=f"blocked_output:{category}",
                message=(
                    "I can't provide operational harmful instructions. "
                    "I can summarize the risk at a high level or suggest defensive mitigations."
                ),
            )
    return GuardrailResult(allowed=True, action="allowed")
