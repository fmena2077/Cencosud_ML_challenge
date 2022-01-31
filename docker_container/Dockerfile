FROM python:3.8-slim-buster

LABEL Maintainer="Francisco Mena"

ENV DATE=2019-12-13

WORKDIR /usr/app/src

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENTRYPOINT [ "python", "./inference.py"]



