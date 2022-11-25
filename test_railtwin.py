# 20.06.2022----------------------------------------------------------------------------------------------------------------------
#  created by: felix 
#              felix@eickeler.com       
# ----------------------------------------------------------------------------------------------------------------
#

import argparse
import os
import signal
from pathlib import Path
from subprocess import check_output

import torch

from railtwin_common.common.io import read
from railtwin_common.common.io.io_options import IOOptions
from railtwin_common.common.io.read_router import scandir_supported
from railtwin_common.common.io.writer.ply import ply_writer
from railtwin_common.common.logger import RailTwinLogger
from utils.config_railtwin import RailTwinConfig as Config
from utils.predictor import Predictor

logger = RailTwinLogger.create()


def setup_cfg(cfg_path, chkps_path):
    # Initialize configuration class
    # cfg = RailTwinConfig()
    cfg = Config()
    cfg.load(cfg_path)
    # cfg.augment_scale_anisotropic = False
    # cfg.augment_symmetries = [False, False, False]
    cfg.augment_noise = 0
    cfg.augment_color = 0.0
    cfg.saving = False
    cfg.saving_path = "rtib_tmp"
    cfg.chkp_path = chkps_path
    return cfg


def check_gpu_memory(gpu_id):
    smi_call = check_output("nvidia-smi --query-gpu=utilization.gpu,memory.free,memory.used --format=csv", shell=True)
    devices = str(smi_call).split("\\n")[1:-1]
    line = devices[gpu_id]
    gpu, mem, memtot = [int(i.replace(" %", "").replace(" MiB", "")) for i in line.split(",")]
    return gpu, mem, memtot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch from Docker')
    parser.add_argument('-m', '--model', type=Path, help='Path to the model that should be tested')
    parser.add_argument('-i', '--input_path', type=Path, help="Path to the file or folders with files", default="~/results")
    parser.add_argument('-o', '--output_path', type=Path, help="Path to the output folder", default=None)
    parser.add_argument('--metadata', type=Path, help="Checkpoint you want to continue from", default="/default/thing/that/does/not/exist")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-gid', '--gpu_id', default="0", type=str, help="Choose the GPU_ID for training")
    args = parser.parse_args()

    GPU_ID = args.gpu_id
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    if args.input_path.exists():
        if args.input_path.is_dir():
            # get parsable data
            input_paths = scandir_supported(args.input_path)
            if len(input_paths) == 0:
                raise FileNotFoundError("The directory does not contain any inference-able files")
        else:
            input_paths = [args.input_path]
    else:
        raise FileNotFoundError("The given path does not exists")

    if not args.output_path:
        if args.input_path.is_dir():
            output_path = input_paths
        else:
            output_path = input_paths.parent
    else:
        output_path = args.output_path
    output_path.mkdir(exist_ok=True, parents=True)

    model = Path(args.model)
    model_path = None
    if model.is_dir():
        chkps = list(model.glob("chkp_*.tar"))
        chkps.sort(key=lambda x: os.path.getmtime(x))
        if len(chkps) != 0:
            model_path = chkps[-1].resolve()
        cfg_path = model / "parameters.json"

        # just so it works on result folder
        if not cfg_path.exists():
            cfg_path = model.parent / "parameters.json"
    else:
        cfg_path = model.parent / "parameters.json"

    cfg = setup_cfg(cfg_path=cfg_path, chkps_path=model_path)
    cfg.val_batch_num = 20
    cfg.batch_num = 10
    cfg.validation_size = 200

    predictor = Predictor(cfg)
    for file in input_paths:
        try:
            pc = read(file)
        except Exception as e:
            raise e


        gpu, mem, mem_used = check_gpu_memory(gpu_id=int(GPU_ID))
        logger.info(f"Resources used: \t GPU: {gpu}, \tMemFree: {mem}, \tMemAlloc: {mem_used}")
        logger.info(f"Working on {file}")
        predictions = predictor.run_on_cloud(pc, cache_name=file.stem)
        io_options = IOOptions.do_nothing()
        io_options.binary = True
        ply_writer(output_path / f"{file.stem}_predicted.ply", predictions, options=io_options)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)

    # train_railtwin(args.dataset.expanduser(), args.working_dir.expanduser(), args.chkp_path.expanduser())  # , args.config_path.expanduser())
