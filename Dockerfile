FROM ubuntu:latest

WORKDIR /subgen

RUN apt-get update && apt-get -y install python3 python3-pip

ADD https://raw.githubusercontent.com/McCloudS/subgen/main/subgen/subgen.py /subgen/subgen.py

CMD ["python", "-u", "subgen.py"]

EXPOSE 8090
