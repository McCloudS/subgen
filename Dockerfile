# Stage 1: Builder
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 as builder

WORKDIR /subgen

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Stage 2: Runtime
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /subgen

# Copy necessary files from the builder stage
COPY --from=builder /subgen/requirements.txt .
COPY --from=builder /subgen/launcher.py .
COPY --from=builder /subgen/subgen.py .
COPY --from=builder /subgen/language_code.py .
COPY --from=builder /user/local /usr/local

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

# Set command to run the application
CMD ["python3", "launcher.py"]
