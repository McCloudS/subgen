FROM ubuntu:latest

WORKDIR /subgen

RUN apt-get update && apt-get -y install python3 python3-pip

ADD https://raw.githubusercontent.com/McCloudS/subgen/main/subgen/requirements.txt /subgen/requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["/subgen/subgen.py"]

EXPOSE 8090
