FROM ubuntu:19.04

RUN  apt-get update \
  && apt-get install --yes --no-install-recommends \
     build-essential \
     gcc \
     cmake \
     python3-dev \
     python3-setuptools \
     python3-wheel \
     python3-numpy \
  && apt-get clean

# CGAL dependencies
RUN  apt-get update \
  && apt-get install --yes --no-install-recommends \
     wget \
     libboost-dev \
     libboost-program-options-dev \
     libboost-system-dev \
     libboost-thread-dev \
     libgmp10-dev \
     libmpfr-dev \
     zlib1g-dev \
  && apt-get clean

RUN mkdir -p /source \
  && cd /source \
  && apt-get install \
     wget \
     ca-certificates \ 
  && CGAL_VERSION=CGAL-4.13.1 \
  && wget "https://github.com/CGAL/cgal/releases/download/releases%2F${CGAL_VERSION}/${CGAL_VERSION}.tar.xz" \
  && tar xf ${CGAL_VERSION}.tar.xz \
  && rm -v ${CGAL_VERSION}.tar.xz \
  && ln -s ${CGAL_VERSION} cgal \
  && cd cgal \
  && cmake -DCGAL_HEADER_ONLY=ON -DCMAKE_BUILD_TYPE=Release -DWITH_CGAL_Qt5=OFF -DWITH_CGAL_ImageIO=OFF -DCGAL_Boost_USE_STATIC_LIBS=ON .
 
ENV CGAL_DIR /source/cgal

VOLUME /localfs
WORKDIR /localfs
CMD ["/bin/bash"]
