FROM ubuntu:19.04 

RUN apt-get update \
 && apt-get install --yes --no-install-recommends \
    python3 \
    python3-numpy \
    python3-setuptools \
    python3-pip \
 && apt-get clean 

WORKDIR /build

