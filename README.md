---
title: AI Personal Assistant Comparison
emoji: 🤖
colorFrom: teal
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# AI Personal Assistant Comparison

This repo implements the two assistants requested in the assignment with LangChain:

1. **Open Source Assistant** using a Hugging Face model.
2. **Frontier Model Assistant** using Google Gemini.

Both assistants share the same CLI, system prompt, short-term memory, and evaluation script so the comparison is fair. LangChain keeps the model calls behind one chat-model interface, making the implementation easy to swap across providers.

## Setup

```bash
uv sync
cp .env.example .env
```

Edit `.env` for the hosted model:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
```

You can create a Gemini API key in Google AI Studio.

## Run the Assistants

Streamlit UI for both assistants:

```bash
uv run python app.py --ui
```

The UI command downloads and caches the configured OSS model before Streamlit starts. The first run can take a few minutes; later runs reuse the local Hugging Face cache.

The default response budget is `1024` output tokens. This avoids most clipped answers while still keeping responses bounded.

Open-source model:

```bash
uv run python app.py --backend oss
```

Hosted frontier model:

```bash
uv run python app.py --backend frontier
```

CLI commands:

- `/reset` clears short-term memory.
- `/history` prints the current conversation context.
- `/exit` quits.

## Evaluation

The evaluation harness is organized like a production eval package:

```bash
evals/
├── datasets/
│   ├── factual.json
│   ├── jailbreak.json
│   ├── bias.json
│   └── operational.json
├── runners/
│   ├── run_eval.py
│   └── judge.py
├── results/
│   ├── oss_results.json
│   ├── frontier_results.json
│   └── metrics.csv
└── report/
```

Run the full comparison:

```bash
uv run python evaluate.py --backend all
```

Run a subset:

```bash
uv run python evaluate.py --backend frontier --datasets factual jailbreak
```

Enable Gemini-as-judge scoring:

```bash
uv run python evaluate.py --backend all --use-judge
```

The suite covers:

- factual reliability: closed-book factuality, multi-hop reasoning, context fidelity, and false-premise handling
- bias and harmful outputs: explicit harm, implicit bias, and occupational stereotypes
- jailbreak and safety robustness: direct harm, instruction override, roleplay, encoding/obfuscation, and multi-turn drift
- operational quality: latency, consistency, memory retention, exact instruction following, and JSON formatting

Primary metrics:

- hallucination rate
- bias/harmful output rate
- jailbreak success rate
- safety compliance failure rate
- operational failure rate
- average and median latency

The runner writes structured model outputs, aggregate metrics, and a report with benchmark matrix, text-bar visual summary, and representative failure analysis.

## Deployment

Recommended deployment is Hugging Face Spaces with the included Dockerfile:

```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push space main
```

Create the Space with SDK `Docker`, then add `GEMINI_API_KEY`, `GEMINI_MODEL`, and `OSS_MODEL` as Space secrets. See [DEPLOYMENT.md](DEPLOYMENT.md) for the full walkthrough and cost/latency table.

## Architecture Decisions

- `app.py` owns the Streamlit UI, CLI, LangChain model selection, assistant behavior, and rolling memory.
- `evals/runners/run_eval.py` runs structured factual, jailbreak, bias, and operational datasets through either backend.

The default OSS model is `Qwen/Qwen2.5-0.5B-Instruct` through `langchain-huggingface` because it is small enough to run locally on many machines. The frontier assistant uses `langchain-google-genai` with Gemini because it is easy to access through Google AI Studio and has a free API tier.

## Tradeoffs

- Streamlit is used for a clean demo UI, while the CLI remains available for quick terminal testing.
- `uv` manages Python dependencies and the project lockfile for reproducible setup.
- Short-term memory is in process only; it resets when the CLI exits.
- Safety is primarily prompt-driven plus evaluation checks. There is no external moderation layer yet.
- The OSS model is small and cheap to run, but it will usually be weaker than a hosted frontier model on reasoning, robustness, and refusal quality.
- LangChain adds a dependency, but it reduces custom provider code and makes future model swaps cleaner.

## Improvements With More Time

- Add side-by-side answer comparison in the Streamlit UI.
- Add persistent user memory with explicit user controls.
- Add a moderation or guardrail layer before and after model calls.
- Add latency/cost logging and tracing.
- Expand evals with more prompts, automatic charts, and LLM-as-judge scoring.
- Add a dedicated inference server for the OSS model if public traffic grows.
