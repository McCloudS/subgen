# Stage 1: Builder
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS builder

WORKDIR /subgen
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg git tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Stage 2: Runtime
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /subgen

# Copy files
COPY --from=builder /subgen/launcher.py .
COPY --from=builder /subgen/subgen.py .
COPY --from=builder /subgen/language_code.py .
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# --- FIX 1: Install gosu ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg python3 curl gosu \
    && rm -rf /var/lib/apt/lists/*

# --- FIX 2: Create a dedicated cache directory ---
# This prevents the app from trying to write to root-owned /.cache
RUN mkdir -p /cache && chmod 777 /cache

# --- FIX 3: Set Environment Vars to use the new cache ---
# This forces HuggingFace and Matplotlib to write to our writable folder
ENV XDG_CACHE_HOME=/cache
ENV HF_HOME=/cache/huggingface
ENV MPLCONFIGDIR=/cache/matplotlib
ENV PYTHONUNBUFFERED=1

# Copy and enable entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "launcher.py"]
