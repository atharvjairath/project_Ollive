from __future__ import annotations

import modal


APP_NAME = "qwen-oss-assistant-api"
MODEL_CACHE_PATH = "/cache/huggingface"

hf_cache = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch")
    .pip_install_from_requirements("requirements-deploy.txt")
    .env({"HF_HOME": MODEL_CACHE_PATH})
    .add_local_python_source("assistant_service")
    .add_local_python_source("app")
)

app = modal.App(APP_NAME, image=image)


@app.function(
    volumes={MODEL_CACHE_PATH: hf_cache},
    timeout=20 * 60,
)
def download_model() -> None:
    from app import default_model_name, download_oss_model

    download_oss_model(default_model_name("oss"))
    hf_cache.commit()


@app.function(
    volumes={MODEL_CACHE_PATH: hf_cache},
    timeout=20 * 60,
    scaledown_window=10 * 60,
    gpu="T4",
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def fastapi_app():
    from app import app as web_app

    return web_app
