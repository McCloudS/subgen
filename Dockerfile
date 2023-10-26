FROM ubuntu:latest

WORKDIR /subgen

RUN apt-get update && apt-get -y install ffmpeg python3 python3-pip

ENV PYTHONUNBUFFERED 1

ADD https://raw.githubusercontent.com/McCloudS/subgen/main/bootstrap.sh /bootstrap.sh
RUN chmod +x /bootstrap.sh

CMD [ "/bootstrap.sh" ]

EXPOSE 8090
