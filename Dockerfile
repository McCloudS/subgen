FROM ubuntu:latest

WORKDIR /subgen

RUN apt-get update && apt-get -y install python3 python3-pip

RUN pip install -r requirements.txt

ENTRYPOINT ["python3"]
CMD ["/subgen/subgen.py"]

EXPOSE 8090
