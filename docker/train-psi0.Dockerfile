# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/venv \
    PATH="/venv/bin:${PATH}" \
    PYTHONPATH="/workspace/src"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    python3.10 \
    python3.10-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --no-cache-dir uv==0.8.20

WORKDIR /workspace
COPY pyproject.toml uv.lock README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md ./
COPY LICENSES ./LICENSES
COPY src ./src

RUN uv sync --locked --python 3.10 --group psi --group cpu-test --group serve --group robot

COPY . .

CMD ["zeta", "--help"]
