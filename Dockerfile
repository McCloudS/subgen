FROM ubuntu:latest

RUN apt-get update && apt-get -y install python3 python3-pip

ADD https://raw.githubusercontent.com/McCloudS/subgen/main/subgen/subgen.py /subgen

CMD [ "python3", "-u", "/subgen.py"]

EXPOSE 8090
