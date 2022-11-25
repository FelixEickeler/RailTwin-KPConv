#FROM nvidia/cuda:11.0-devel-ubuntu20.04
FROM nvidia/cuda:11.0.3-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV TERM xterm-256color
#
RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/cuda.list && \ apt-key del 7fa2af80 && \ apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
#RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub


RUN apt-get  -y update && apt-get install -y \
	git sudo nano htop gosu openssh-server \
    libpython3.8-dev python3 python3-pip wget libgl-dev \
    python3-tk 
#
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN rm requirements.txt
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# RailTwin Comon Dependecies
RUN pip3 install pyarrow pybind11 spdlog laspy plyfile

# create user, ids are temporary
ARG USER_ID=1000
RUN useradd -m --no-log-init point_warrior && yes slice | passwd point_warrior
RUN usermod -aG sudo point_warrior
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN chown -R point_warrior:sudo /home/point_warrior
#
# Create entrypoint
COPY kpconv-entrypoint.sh /usr/local/bin/kpconv-entrypoint.sh
RUN chmod +x /usr/local/bin/kpconv-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/kpconv-entrypoint.sh"]
#
# Install kpconv-ml
RUN su point_warrior
WORKDIR /home/point_warrior
