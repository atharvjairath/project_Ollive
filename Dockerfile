FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV HF_HOME=/data/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"

COPY requirements-deploy.txt ./
RUN uv venv \
    && uv pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install -r requirements-deploy.txt

COPY . .

EXPOSE 7860

CMD ["uv", "run", "python", "app.py", "--ui", "--server-name", "0.0.0.0", "--server-port", "7860"]
