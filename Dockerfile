FROM nvidia/cuda:12.3.2-base-ubuntu22.04

COPY requirements.txt entrypoint.sh /

# --- FIX 1: Install gosu ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg python3 python3-pip curl gosu tzdata \
    && python3 -m pip install -U --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && python3 -m pip install -U --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove python3-pip \
    && rm -rf \
    /var/lib/apt/lists/* \
    /tmp/*

WORKDIR /subgen

# Copy files
COPY launcher.py subgen.py language_code.py /subgen/

# --- FIX 2: Create a dedicated cache directory ---
# This prevents the app from trying to write to root-owned /.cache
RUN mkdir -p /cache && chmod 777 /cache

# --- FIX 3: Set Environment Vars to use the new cache ---
# This forces HuggingFace and Matplotlib to write to our writable folder
ENV XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    MPLCONFIGDIR=/cache/matplotlib \
    PYTHONUNBUFFERED=1 

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "launcher.py"]
