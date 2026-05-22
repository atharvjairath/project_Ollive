---
title: Qwen2.5-0.5B-Instruct
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Qwen2.5-0.5B-Instruct

Public OSS model endpoint for the assistant assignment.

## API

Health check:

```bash
curl https://atharvjairath-qwen2-5-0-5b-instruct.hf.space/health
```

Generate:

```bash
curl -X POST https://atharvjairath-qwen2-5-0-5b-instruct.hf.space/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Say hello in one short sentence."}'
```

The response includes generated text, latency, estimated output tokens, and estimated tokens/sec.
