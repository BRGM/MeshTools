FROM ubuntu:18.10 AS builder
RUN  apt-get update \
  && apt-get install --yes --no-install-recommends \
     build-essential \
     gcc \
     cmake \
     libcgal-dev \
     python3-dev \
     python3-setuptools \
     python3-wheel \
     python3-numpy \
  && apt-get clean

WORKDIR /build
