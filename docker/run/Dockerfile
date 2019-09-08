FROM ubuntu:19.04 

RUN apt-get update \
 && apt-get install --yes --no-install-recommends \
    python3 \
    python3-numpy \
    python3-setuptools \
    python3-pip \
    python3-wheel \
    python3-pytest-xdist \
    libcgal13 \
 && apt-get clean 

VOLUME /localfs
WORKDIR /localfs
CMD ["/bin/bash"]
