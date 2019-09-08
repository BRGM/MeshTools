FROM ubuntu:19.04

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

ENV CGAL_DIR /usr/include/CGAL

VOLUME /localfs
WORKDIR /localfs
CMD ["/usr/bin/bash"]
