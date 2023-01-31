FROM python

RUN apt-get update && apt-get install -y ffmpeg git gcc

RUN pip3 install flask requests

ADD https://raw.githubusercontent.com/McCloudS/subgen/main/subgen/subgen.py /

CMD [ "python3", "-u", "/subgen.py"]
