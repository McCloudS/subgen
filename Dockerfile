FROM ubuntu:latest

RUN apt-get update && apt-get install python3 python-pip

ADD https://raw.githubusercontent.com/McCloudS/subgen/main/subgen/subgen.py /

CMD [ "python3", "-u", "/subgen.py"]

EXPOSE 8090
