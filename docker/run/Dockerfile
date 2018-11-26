FROM debian:buster-slim

RUN apt-get update \
 && apt-get install --yes --no-install-recommends \
    python3.7 \
    python3-numpy \
     python3-setuptools \
 && apt-get clean 

WORKDIR /build

# We install pip from sources as python3-pip package uses python3.6
RUN apt-get install --yes --no-install-recommends \
    wget \
 && apt-get clean \
 && wget --no-check-certificate https://files.pythonhosted.org/packages/45/ae/8a0ad77defb7cc903f09e551d88b443304a9bd6e6f124e75c0fbbf6de8f7/pip-18.1.tar.gz \
 && tar xf pip-18.1.tar.gz \
 && cd pip-18.1 \
 && python3.7 setup.py install \
 && cd .. \
 && rm -rf pip-18.1*

 
