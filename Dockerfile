FROM nvidia/cuda:12.3.2-base-ubuntu22.04

# Apt packages — own layer so pip changes don't re-run apt
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg python3 python3-pip curl gosu tzdata

# Torch — large and rarely changes; own layer so requirements.txt changes don't bust it
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -U torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# App dependencies — only rebuilds when requirements.txt changes
COPY requirements.txt /
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -U -r /requirements.txt \
    && apt-get purge -y --auto-remove python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /subgen

# App files last — changes here don't bust the layers above
COPY launcher.py subgen.py language_code.py /subgen/

RUN mkdir -p /cache && chmod 777 /cache

ENV XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    MPLCONFIGDIR=/cache/matplotlib \
    PYTHONUNBUFFERED=1

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "launcher.py"]
