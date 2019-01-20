FROM debian:stable AS builder
RUN  apt-get update \
  && apt-get install --yes --no-install-recommends \
     build-essential \
     gcc \
     cmake \
     wget \
     # cgal dependencies - begin \
     libboost-dev \
     libboost-system-dev \
     libboost-thread-dev \
     libgmp10-dev \
     libmpfr-dev \
     libmpfi-dev \
     # cgal dependencies - end \
     python3-dev \
     python3-setuptools \
     python3-wheel \
     python3-numpy \
  && apt-get clean
RUN mkdir -p /build/thirdparties \
 && cd /build/thirdparties \
 && wget --no-check-certificate https://github.com/CGAL/cgal/releases/download/releases/CGAL-4.13/CGAL-4.13.tar.xz \
 && tar xf CGAL-4.13.tar.xz \
 && mv CGAL-4.13 cgal
 
WORKDIR /build
