FROM ubuntu:22.04

WORKDIR /home

ENV SHELL /bin/bash

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
    python3-tk \
    coinor-libipopt-dev \
    wget \
    gdb

# Numpy 1.26.4 is neededbecause numpy
# 2.X had breaking changes for quadruped_drake.
# Drake 1.24.0 is the newest version that works
# with quadruped_drake
RUN pip3 install \
    numpy==1.26.4 \
    drake==1.24.0 \
    scipy==1.14.1 \
    matplotlib \
    shapely

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
#     cmake .. -DCMAKE_BUILD_TYPE=Debug && \
#     make

RUN cd repos && \
    apt-get install -y --no-install-recommends ca-certificates gnupg lsb-release wget && \
    wget -qO- https://drake-apt.csail.mit.edu/drake.asc | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/drake.gpg >/dev/null && \
    echo "deb [arch=amd64] https://drake-apt.csail.mit.edu/$(lsb_release -cs) $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/drake.list >/dev/null && \
    apt update && \
    apt install -y --no-install-recommends drake-dev && \
    export PATH="/opt/drake/bin${PATH:+:${PATH}}" && \
    export PYTHONPATH="/opt/drake/lib/python$(python3 -c 'import sys; print("{0}.{1}".format(*sys.version_info))')/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
