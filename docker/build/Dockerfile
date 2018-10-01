From debian:buster as builder
RUN  apt-get update && apt-get install --yes build-essential gcc cmake libcgal-dev python3 python3-dev
# we are using pip to build and install the package...
# numpy should be in the run environment
RUN apt-get install --yes python3-pip python3-numpy
RUN apt-get clean
WORKDIR /build/
ENTRYPOINT ["/bin/bash"]

