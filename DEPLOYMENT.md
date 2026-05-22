# Deployment Guide

## Recommended Path: Hugging Face Spaces

This satisfies the bonus requirement to deploy the OSS model publicly. The app runs as a Docker-based Streamlit Space and downloads `Qwen/Qwen2.5-0.5B-Instruct` before serving the UI.

## Steps

1. Create a new Hugging Face Space.
   - SDK: Docker
   - Visibility: Public
   - Hardware: CPU basic is enough for a demo, but it will be slow. Upgrade hardware if latency matters.

2. Push this repo to the Space.

```bash
git init
git add .
git commit -m "Deploy assistant comparison app"
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push space main
```

3. Add secrets in Space settings.

```text
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
OSS_MODEL=Qwen/Qwen2.5-0.5B-Instruct
```

`GEMINI_API_KEY` is optional if you only demo the OSS assistant, but required for the Gemini assistant and judge evals.

4. Wait for the Space build to finish.

The first startup downloads the OSS model into the Hugging Face cache. Later restarts reuse the cache when persistent storage is available.

The Dockerfile uses `requirements-deploy.txt` and installs CPU-only PyTorch first to avoid pulling CUDA wheels on Hugging Face Spaces CPU hardware. Local development still uses `uv.lock`.

## Expected Cost and Latency

| Deployment | Cost | Expected Latency | Notes |
|---|---:|---:|---|
| Hugging Face Spaces CPU Basic | Free tier when available | Slow for OSS inference, often several seconds per answer | Good for public demo |
| Hugging Face Spaces upgraded CPU/GPU | Paid | Lower latency | Better for interviews/demo reliability |
| Gemini API | Free tier available, quota-limited | Usually faster than local OSS on CPU | Requires secret key |

## Production Improvements

- Add persistent storage for model cache.
- Add health checks and startup progress display.
- Add request logging and latency tracing.
- Add rate limits for public usage.
- Split OSS inference into a dedicated model server if traffic increases.
