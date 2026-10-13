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


def setup_cfg(cfg_path, chkps_path, drop_features=False):
    # Initialize configuration class
    # cfg = RailTwinConfig()
    # load() fills the class from the file and returns a fresh instance (derived values like num_layers recomputed)
    cfg = Config.load(cfg_path)
    # cfg.augment_scale_anisotropic = False
    # cfg.augment_symmetries = [False, False, False]
    cfg.augment_noise = 0
    # Zero the input features (e.g. colour / intensity) for every point, see datasets/features.py
    cfg.drop_features = drop_features
    cfg.saving = False
    cfg.saving_path = "rtib_tmp"
    cfg.chkp_path = chkps_path
    return cfg


def check_gpu_memory(gpu_id):
    """(utilisation %, free MiB, used MiB) of a GPU, None if nvidia-smi is not available."""
    try:
        smi_call = check_output("nvidia-smi --query-gpu=utilization.gpu,memory.free,memory.used "
                                "--format=csv,noheader,nounits", shell=True)
        line = smi_call.decode().strip().splitlines()[gpu_id]
        return tuple(int(i) for i in line.split(","))
    except Exception:
        return None


def resolve_model(model):
    """
    Checkpoint and parameters.json of a model given as run folder, checkpoints folder or .tar file.
    Newest chkp_*.tar wins, current_chkp.tar is the fallback.
    """
    model = Path(model).expanduser()
    if model.is_file():
        model_path = model.resolve()
        folders = [model.parent, model.parent.parent]
    else:
        candidates = [model, model / 'checkpoints']
        chkps = sorted([c for d in candidates for c in d.glob('chkp_*.tar')], key=os.path.getmtime)
        if not chkps:
            chkps = [c for d in candidates for c in d.glob('current_chkp.tar')]
        if not chkps:
            raise FileNotFoundError(f"No checkpoint (chkp_*.tar / current_chkp.tar) found in {model}")
        model_path = chkps[-1].resolve()
        folders = [model, model.parent]
    cfg_path = next((f / 'parameters.json' for f in folders if (f / 'parameters.json').exists()), None)
    if cfg_path is None:
        raise FileNotFoundError(f"No parameters.json found next to {model}")
    return model_path, cfg_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch from Docker')
    parser.add_argument('-m', '--model', type=Path, help='Path to the model that should be tested')
    parser.add_argument('-i', '--input_path', type=Path, help="Path to the file or folders with files", default="~/results")
    parser.add_argument('-o', '--output_path', type=Path, help="Path to the output folder", default=None)
    parser.add_argument('--metadata', type=Path, help="Checkpoint you want to continue from", default="/default/thing/that/does/not/exist")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-gid', '--gpu_id', default="0", type=str, help="Choose the GPU_ID for training")
    parser.add_argument('--drop-features', action='store_true',
                        help="Predict without input features (zeroed), e.g. for data without colour / intensity")
    args = parser.parse_args()

    GPU_ID = args.gpu_id
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    input_path = args.input_path.expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"The given path does not exist: {input_path}")
    if input_path.is_dir():
        # get parsable data, without the outputs of earlier runs
        input_paths = [p for p in scandir_supported(input_path) if not Path(p).stem.endswith('_predicted')]
        if len(input_paths) == 0:
            raise FileNotFoundError("The directory does not contain any inference-able files")
        input_root = input_path
    else:
        input_paths = [input_path]
        input_root = input_path.parent

    # Default: next to the inputs. Subfolders are mirrored, so files with the same name do not collide.
    output_root = args.output_path.expanduser() if args.output_path else input_root

    model_path, cfg_path = resolve_model(args.model)
    logger.info(f"Model: {model_path}")

    cfg = setup_cfg(cfg_path=cfg_path, chkps_path=model_path, drop_features=args.drop_features)
    cfg.val_batch_num = 20
    cfg.batch_num = 10
    cfg.validation_size = 200

    predictor = Predictor(cfg)
    for file in input_paths:
        file = Path(file)
        relative = file.relative_to(input_root)
        try:
            pc = read(file)
        except Exception as e:
            logger.error(f"Could not read {file}: {e}")
            continue

        resources = check_gpu_memory(gpu_id=int(GPU_ID.split(',')[0]))
        if resources:
            logger.info(f"Resources used: \t GPU: {resources[0]} %, \tMemFree: {resources[1]} MiB, "
                        f"\tMemUsed: {resources[2]} MiB")
        logger.info(f"Working on {file}")
        # Cache name unique per input file (relative path) and model checkpoint
        cache_name = "_".join(relative.with_suffix('').parts)
        predictions = predictor.run_on_cloud(pc, cache_name=cache_name)
        io_options = IOOptions.do_nothing()
        io_options.binary = True
        output_file = output_root / relative.parent / f"{file.stem}_predicted.ply"
        output_file.parent.mkdir(exist_ok=True, parents=True)
        ply_writer(output_file, predictions, options=io_options)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)

    # train_railtwin(args.dataset.expanduser(), args.working_dir.expanduser(), args.chkp_path.expanduser())  # , args.config_path.expanduser())
