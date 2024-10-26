FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

WORKDIR /subgen

ADD https://raw.githubusercontent.com/McCloudS/subgen/main/requirements.txt /subgen/requirements.txt

RUN apt-get update \
    && apt-get install -y \
        python3 \
        python3-pip \
        ffmpeg \
        deep-translator \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install -r requirements.txt

RUN pip install googletrans==4.0.0-rc1

ENV PYTHONUNBUFFERED=1

ADD https://raw.githubusercontent.com/McCloudS/subgen/main/launcher.py /subgen/launcher.py
ADD https://raw.githubusercontent.com/McCloudS/subgen/main/subgen.py /subgen/subgen.py

CMD [ "bash", "-c", "python3 -u launcher.py" ]
