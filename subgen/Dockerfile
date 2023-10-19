FROM python

RUN pip3 install flask requests stable-ts fast-whisper

ADD https://raw.githubusercontent.com/McCloudS/subgen/main/subgen/subgen.py /

CMD [ "python3", "-u", "/subgen.py"]
