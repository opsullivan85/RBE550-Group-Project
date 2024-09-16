FROM robotlocomotion/drake:noble-1.32.0

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

# Update and upgrade the system
RUN apt-get update && apt-get upgrade -y

WORKDIR /home

ENV SHELL /bin/bash

# Set the default command to bash
CMD ["/bin/bash"]

RUN apt install git-all -y && \
    apt install cmake -y && \
    apt install clang -y

# TODO install quadruped drake dependencies

RUN mkdir repos && \
    cd repos && \
    git clone https://github.com/vincekurtz/quadruped_drake.git --depth 1
    cd quadruped_drake && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make

# Switch to the created user
USER $USERNAME