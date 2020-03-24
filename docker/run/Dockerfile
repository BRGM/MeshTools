FROM ubuntu:19.04

RUN apt-get update \
 && apt-get install --yes --no-install-recommends \
    python3 \
    python3-numpy \
    python3-setuptools \
    python3-pip \
    python3-wheel \
    libgmp10 \
    libmpfr6 \
 && apt-get clean

# AttributeError: 'Function' object has no attribute 'get_marker'
# with earlier versions
# e.g. https://github.com/pytest-dev/pytest-cov/issues/252
RUN pip3 install pytest==3.10.1

VOLUME /localfs
WORKDIR /localfs
CMD ["/bin/bash"]
