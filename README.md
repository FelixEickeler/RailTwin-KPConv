

![Intro figure](doc/2022-09-22_16_20_synth.png)

Modified by Felix Eickeler, Created by Hugues THOMAS, 

## Introduction

This is a modified version of the KPConv-Pytorch git repository [1], adapted to conduct experiments for the detection of railways.
We appreciate the effort and work that Thomas Hugues has put into the original project in code, help and documentation.<br/>
The git was used as detection component for the paper **Enhancing Railway Detection by Priming Neural Networks with Project Exaptations**. [2]


Mostly the modifications consist of 3 parts: 
- data + config loading
- custom training
- result writing

Most of the dataset splits and types can be change by modifying the import statements. 
Files are usually named rtib for dataset related tasks or RailTwin for generic use.

## Getting Started
Make sure that the common submodule is pulled:
```bash
git submodule init
git submodule update
```

This git is designed to be used in a docker (Ubuntu 22.04, Python 3.10, PyTorch 2.3.1, CUDA 12.1) and connected to
over an ssh connection or `docker compose exec`:
```bash
cd docker
cp .env.example .env      # set DATA_DIR, WORK_DIR, LOCAL_UID / LOCAL_GID (id -u / id -g) and the ssh password (or public key)
docker compose up -d --build
docker compose exec kpconv bash          # or: ssh -p 2244 point_warrior@<host>
```
- The dataset folder is mounted at `~/dataset`, the work folder at `~/workdir`, and this repository at
  `~/railtwin_kpconv`.
- The C++ wrappers are compiled on the first start.
- TensorBoard runs on host port 6667.
- PyCharm: SSH interpreter `point_warrior@<host>:2244`, `/usr/bin/python3`, path mapping
  `<repo> -> /home/point_warrior/railtwin_kpconv` (the repository is mounted, no upload needed on the same host).
- The first build takes some time.

Without Docker: `pip install -r docker/requirements.txt`, then `cd cpp_wrappers && bash compile_wrappers.sh`.
Tests: `python -m pytest tests/`.

## Experiments

We are currently working on pubilshing the rtib+ dataset.
However the datasets given, are only partially available (see [2])

## Sources
[1] https://github.com/HuguesTHOMAS/KPConv-PyTorch

[2] https://www.mdpi.com/2072-4292/14/21/5482

