# syntax=docker/dockerfile:1
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/venv_lerobot/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir uv==0.8.20

WORKDIR /workspace
COPY . .

RUN uv venv --python 3.12 /venv_lerobot \
    && uv pip install --python /venv_lerobot/bin/python -r real/g1/scripts/requirements-lerobot.txt \
    && uv pip install --python /venv_lerobot/bin/python --no-deps -e real/g1/adapters/lerobot_robot_sonic

CMD ["python", "real/g1/run.py", "--help"]
