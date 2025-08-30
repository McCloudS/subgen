# Stage 1: Builder
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS builder

WORKDIR /subgen

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Stage 2: Runtime
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /subgen

# Copy necessary files from the builder stage
COPY --from=builder /subgen/launcher.py .
COPY --from=builder /subgen/subgen.py .
COPY --from=builder /subgen/language_code.py .
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

# Set command to run the application
CMD ["python3", "launcher.py"]
