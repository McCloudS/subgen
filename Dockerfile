FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /subgen

RUN apt-get update \
    && apt-get install -y \
        python3 \
        python3-pip \
        ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install \
    numpy \
    stable-ts \
    fastapi \
    requests \
    faster-whisper \
    uvicorn \
    python-multipart \
    python-ffmpeg \
    whisper \
    transformers \
    accelerate \
    optimum

ADD https://raw.githubusercontent.com/McCloudS/subgen/main/launcher.py /subgen/launcher.py
ADD https://raw.githubusercontent.com/McCloudS/subgen/main/subgen/subgen.py /subgen/subgen.py

CMD [ "bash", "-c", "python3 -u launcher.py && python3 -u subgen.py" ]
