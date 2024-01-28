FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /subgen

RUN apt-get update && apt-get -y install python3 python3-pip ffmpeg
RUN pip3 install numpy stable-ts fastapi requests faster-whisper uvicorn python-multipart whisper

ENV PYTHONUNBUFFERED 1

ADD https://raw.githubusercontent.com/McCloudS/subgen/main/subgen/subgen.py /subgen/subgen.py

CMD [ "python3", "-u", "./subgen.py" ]

EXPOSE 8090
