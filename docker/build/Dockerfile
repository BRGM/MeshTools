FROM debian:buster AS builder
RUN  apt-get update \
  && apt-get install --yes --no-install-recommends \
     build-essential \
     gcc \
     cmake \
     libcgal-dev \
     python3.7-dev \
     python3-setuptools \
     python3-wheel \
     python3-numpy \
  && apt-get clean
WORKDIR /build
