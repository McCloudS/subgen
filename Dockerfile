FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

WORKDIR /subgen

RUN apt-get update && apt-get -y install python3 python3-pip ffmpeg

ENV PYTHONUNBUFFERED 1

ADD https://raw.githubusercontent.com/McCloudS/subgen/main/subgen/subgen.py /subgen/subgen.py

CMD [ "python3", "-u", "./subgen.py" ]

EXPOSE 8090
