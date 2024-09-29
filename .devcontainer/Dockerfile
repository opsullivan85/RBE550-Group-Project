FROM ubuntu:22.04

WORKDIR /home

ENV SHELL /bin/bash

CMD ["/bin/bash"]

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
    git-all=1:2.34.1-1ubuntu1.11 \
    cmake=3.22.1-1ubuntu1.22.04.2 \
    clang=1:14.0-55~exp2 \
    build-essential=12.9ubuntu3 \
    libglib2.0-dev=2.72.4-0ubuntu2.3 \
    libeigen3-dev=3.4.0-2ubuntu2 \
    python3-dev=3.10.6-1~22.04.1 \
    python3-pip=22.0.2+dfsg-1ubuntu0.4 \
    coinor-libipopt-dev=3.11.9-2.2build5

# Numpy 1.26.4 is neededbecause numpy
# 2.X had breaking changes for quadruped_drake.
# Drake 1.24.0 is the newest version that works
# with quadruped_drake
RUN pip3 install \
    numpy==1.26.4 \
    drake==1.24.0

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

# # Quadruped drake
# RUN cd repos && \
#     git clone https://github.com/vincekurtz/quadruped_drake.git && \
#     cd quadruped_drake && \
#     git checkout 2b0ba84 && \
#     rm -r .git && \
#     mkdir -p build && \
#     cd build && \
#     cmake .. && \
#     make
