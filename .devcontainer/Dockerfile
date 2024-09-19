# FROM robotlocomotion/drake:noble-1.32.0
FROM ubuntu:22.04

WORKDIR /home

ENV SHELL /bin/bash

# Set the default command to bash
CMD ["/bin/bash"]

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
    git-all \
    cmake \
    clang \
    build-essential \
    libglib2.0-dev \
    libeigen3-dev \
    python3-dev \
    python3-pip \
    coinor-libipopt-dev 

RUN pip3 install \
    numpy==1.26.4 \
    drake==1.8.0

# IFOPT
RUN mkdir repos && \
    cd repos && \
    git clone https://github.com/ethz-adrl/ifopt.git && \
    cd ifopt && \
    git checkout abcf7fc && \
    rm -r .git && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install

# LCM
RUN cd repos && \
    git clone https://github.com/lcm-proj/lcm.git && \
    cd lcm && \
    git checkout f18f8a6 && \
    rm -r .git && \
    pip3 install -v . && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install

# Quadruped drake
RUN cd repos && \
    git clone https://github.com/vincekurtz/quadruped_drake.git && \
    cd quadruped_drake && \
    git checkout 2b0ba84 && \
    rm -r .git && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make
