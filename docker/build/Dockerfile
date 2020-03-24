FROM ubuntu:19.10

# The following two lines are necessary for the configuration of tzdata
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# We need python3-dev to have dynamic libraries at runtime
RUN apt-get update && apt-get install --yes --no-install-recommends \
    wget \
    unzip \
    python3-dev \
    python3-click \
    python3-numpy \
    python3-setuptools \
    python3-pytest-xdist \
    python3-pip \
    python3-wheel \
    git \
    build-essential \
    cmake \
    gcc \
    clang-format-9 \
    libboost-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libgmp10-dev \
    libmpfr-dev \
    zlib1g-dev \
    libeigen3-dev \
    python3-sphinx \
    python3-sphinx-rtd-theme \
    pandoc \
    texlive \
 && apt-get clean \
 && python3 -m pip install \
    pre-commit \
    scikit-build \
    recommonmark

RUN mkdir -p /local/src
WORKDIR /local/src
RUN git clone -b releases/CGAL-5.0.2 --single-branch https://github.com/CGAL/cgal.git \
 && rm -rf /local/src/cgal/.git
ENV CGAL_DIR=/local/src/cgal

# install pre-commit environment
WORKDIR /dummy
COPY .pre-commit-config.yaml .
RUN ln -s /usr/bin/clang-format-9 /usr/bin/clang-format \
 && git init \
 && pre-commit install \
 && pre-commit run --all \
 && rm -rf /dummy

RUN mkdir -p /localfs
VOLUME /localfs
WORKDIR /localfs
