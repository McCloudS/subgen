FROM python

RUN apt-get update && apt-get install -y ffmpeg git gcc

RUN pip3 install webhook_listener
EXPOSE 8090

COPY ./subgen/subgen.py /

CMD [ "python3", "-u", "/subgen.py"]
