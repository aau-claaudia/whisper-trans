FROM python:3.10.8-bullseye


MAINTAINER Pelle Rosenbeck Gøeg <prg@its.aau.dk>

RUN apt-get update
RUN apt-get install -y ffmpeg
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

RUN pip install git+https://github.com/openai/whisper.git


RUN whisper --model base dummy.wav; exit 0
RUN whisper --model small dummy.wav; exit 0
RUN whisper --model medium dummy.wav; exit 0
RUN whisper --model large dummy.wav; exit 0


COPY . /app

ENV TZ Europe/Copenhagen

WORKDIR /app
