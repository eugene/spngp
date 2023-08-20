FROM ubuntu:20.04

WORKDIR /app

ENV TZ="America/Sao_Paulo"

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update -y \
    && dpkg --configure -a &&  apt install -y python3-pip libpq-dev python-dev ant make

COPY makefile makefile
COPY datasets datasets
COPY cccp-spngp.py .
COPY concrete-spngp.py .
COPY energy-spngp.py .
COPY learnspngp.py .
COPY spngp.py .

RUN pip3 install --upgrade pip \
    && pip3 install -r datasets/requirements.txt 