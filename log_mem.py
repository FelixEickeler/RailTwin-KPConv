import time
from subprocess import check_output
import threading
import pandas as pd

interval = 1
data = []
init = {
    "start": time.time(),
    "watch_id": 1,
    "max_gpu": 0,
    "max_mem": 1e6,
    "max_memtot": 0,
}


def update():
    global init
    smi_call = check_output("nvidia-smi --query-gpu=utilization.gpu,memory.free,memory.used --format=csv", shell=True)
    devices = str(smi_call).split("\\n")[1:-1]
    line = devices[init["watch_id"]]
    gpu, mem, memtot = [int(i.replace(" %", "").replace(" MiB", "")) for i in line.split(",")]
    # data.append([time.time() - init["start"], gpu, mem])
    if init["max_gpu"] < gpu:
        init["max_gpu"] = gpu
    if init["max_mem"] > mem:
        init["max_mem"] = mem
    if init["max_memtot"] < memtot:
        init["max_memtot"] = memtot

    print(f'id: {init["watch_id"]} -- gpu[%]: {init["max_gpu"]},'
          f'\t mem_free[MiB]: {init["max_mem"]})'
          f'\t mem[MiB]: {init["max_memtot"]})\r')


def startTimer():
    threading.Timer(interval, startTimer).start()
    update()


if __name__ == '__main__':
    startTimer()
