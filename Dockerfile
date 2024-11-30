FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /subgen

ADD https://raw.githubusercontent.com/McCloudS/subgen/main/requirements.txt /subgen/requirements.txt

RUN apt-get update \
    && apt-get install -y \
        python3 \
        python3-pip \
        ffmpeg \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install -r requirements.txt

ENV PYTHONUNBUFFERED=1

ADD https://raw.githubusercontent.com/McCloudS/subgen/main/launcher.py /subgen/launcher.py
ADD https://raw.githubusercontent.com/McCloudS/subgen/main/subgen.py /subgen/subgen.py
ADD https://raw.githubusercontent.com/McCloudS/subgen/main/language_code.py /subgen/language_code.py

CMD [ "bash", "-c", "python3 -u launcher.py" ]
