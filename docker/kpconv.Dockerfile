# RailTwin-KPConv: Ubuntu 22.04, Python 3.10, PyTorch 2.3.1 with CUDA 12.1 (RTX 30xx / 40xx)
# The PyTorch wheels bring their own CUDA libraries, the base image only provides the driver integration.
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    TERM=xterm-256color \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-tk build-essential \
        git sudo nano htop gosu openssh-server libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade pip \
    && pip3 install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# User point_warrior; uid / gid are adapted to the host at start (see entrypoint). Password / keys for ssh are set at start
# from docker/.env, sudo without password for development inside the container.
RUN useradd -m -s /bin/bash --no-log-init point_warrior \
    && usermod -aG sudo point_warrior \
    && echo 'point_warrior ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/point_warrior \
    && chmod 440 /etc/sudoers.d/point_warrior \
    && mkdir -p /var/run/sshd \
    && sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config

COPY kpconv-entrypoint.sh /usr/local/bin/kpconv-entrypoint.sh
RUN chmod +x /usr/local/bin/kpconv-entrypoint.sh

WORKDIR /home/point_warrior/railtwin_kpconv
ENTRYPOINT ["/usr/local/bin/kpconv-entrypoint.sh"]
CMD ["bash"]
