# FROM robotlocomotion/drake:noble-1.32.0
FROM ubuntu:22.04

# Set up build arguments for the username, UID, and GID
ARG USERNAME=USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Remove existing user with the same UID if present
RUN if id -u $USER_UID >/dev/null 2>&1; then \
    userdel $(id -un $USER_UID); \
    fi

# Create a new user with sudo privileges
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    apt-get update && \
    apt-get install -y sudo && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

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
    git clone https://github.com/ethz-adrl/ifopt.git --depth 1 && \
    cd ifopt && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install

# LCM
RUN cd repos && \
    git clone https://github.com/lcm-proj/lcm.git --depth 1 && \
    cd lcm && \
    pip3 install -v . && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install

# Quadruped drake
RUN cd repos && \
    git clone https://github.com/vincekurtz/quadruped_drake.git --depth 1 && \
    cd quadruped_drake && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make

# Switch to the created user
USER $USERNAME