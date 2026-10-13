#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
# ----------------------------------------------------------------------------------------------------------------------
#      Callable script to start a training on RailTwin dataset
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#      Eickeler FELIX - 04/04/2022
#
# ----------------------------------------------------------------------------------------------------------------------
#   Imports and global variables

import os
# Common libs
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from subprocess import check_output
import numpy as np
import argparse

# sys.path.append("/home/point_warrior/KPConv-PyTorch")
# sys.path.append(Path(__file__).parent)

# Dataset
from torch.utils.data import DataLoader
from models.architectures import KPFCNN

# from custom_datasets.railtwin_ecppm2022 import *
# from datasets.S3DIS import S3DISDataset as RailTwinDatasetECPPM2022
# from datasets.RailTwin_Ecppm2022 import RailTwinDatasetECPPM2022
# from datasets.S3DIS import S3DISDataset as RailTwinDatasetECPPM2022
# from datasets.rtib3p import RTIB3p as RailTwinDatasetECPPM2022
from datasets.rtib3p_16classes_add_3 import RTIB3p as RailTwinDatasetECPPM2022
from datasets.rtib3p import RTIB3pSampler as RailTwinSampler
from datasets.rtib3p import RTIB3pCollate as RailTwinCollate
from utils.trainer_railtwin import ModelTrainer
from utils.config_railtwin_tl import RailTwinConfig


# from train_S3DIS import S3DISConfig as RailTwinConfig


def find_checkpoint(path):
    """Return the checkpoint file itself, or the newest *.tar in a run folder (or its checkpoints/ subfolder)."""
    path = Path(path)
    if path.is_file():
        return path.resolve()
    chkp_dir = path / 'checkpoints' if (path / 'checkpoints').is_dir() else path
    chkps = sorted(chkp_dir.glob('*.tar'), key=os.path.getmtime)
    return chkps[-1].resolve() if chkps else None


def train_railtwin(working_path, previous_training_path=Path(''), pretrained_path=None, gid=None):
    """
    :param working_path: output folder of this run
    :param previous_training_path: run folder or checkpoint to resume (weights, optimizer and epoch are restored)
    :param pretrained_path: run folder or checkpoint of the pre-training. Only the weights are loaded, the fine-tuning
                            starts at epoch 0 and follows config.unfreeze_schedule.
    :param gid: CUDA_VISIBLE_DEVICES to use, None = pick free GPUs
    """

    ############################
    # Initialize the environment
    ############################
    working_path.mkdir(exist_ok=True, parents=True)
    tb_logdir = None

    if not gid:
        # Set which gpu is going to be used & Set GPU visible device
        gpu_ids = []
        # for _id in range(1, torch.cuda.device_count()):
        #     gpu_id = _id - 1
        smi_call = check_output("nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv", shell=True)
        devices = str(smi_call).split("\\n")[1:-1]
        for gpu_id, line in enumerate(devices):
            gpu, mem = [int(i.replace(" %", "")) for i in line.split(",")]
            if gpu < 50 and mem < 50:
                print(f"Current GPU usage: {gpu} % and {mem} MB")
                gpu_ids.append(str(gpu_id))

        if len(gpu_ids) == 0:
            raise RuntimeError("No free gpu was found")
        GPU_ID = ",".join(gpu_ids)
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gid

    ############
    # Checkpoint
    ############
    finetune = pretrained_path is not None
    if finetune:
        chosen_chkp = find_checkpoint(pretrained_path)
        if chosen_chkp is None:
            raise FileNotFoundError(f"No checkpoint found in {pretrained_path}")
        print(f"Fine-tuning from: {chosen_chkp}")
    elif previous_training_path.exists():
        chosen_chkp = find_checkpoint(previous_training_path)
        if chosen_chkp is not None:
            print(f"Continuing from: {chosen_chkp}")
    else:
        chosen_chkp = None

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initialize configuration class
    config = RailTwinConfig()

    # Resuming: use the configuration of that run. Fine-tuning keeps this script's configuration.
    if chosen_chkp is not None and not finetune:
        run_dir = previous_training_path if previous_training_path.is_dir() else previous_training_path.parent.parent
        for name in ['parameters.json', 'parameters.txt']:
            if (run_dir / name).exists():
                config = RailTwinConfig.load(run_dir / name)
                break
        config.saving_path = None

    # Get path from argument if given
    if working_path.exists():
        config.saving_path = str(working_path)

    # Initialize datasets
    training_dataset = RailTwinDatasetECPPM2022(config, set='training', use_potentials=True)
    test_dataset = RailTwinDatasetECPPM2022(config, set='validation', use_potentials=True)
    # test_dataset = RailTwinDatasetECPPM2022(config, set='test', use_potentials=True)

    # Initialize samplers
    training_sampler = RailTwinSampler(training_dataset)
    test_sampler = RailTwinSampler(test_dataset)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=RailTwinCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=RailTwinCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    untouched_ratio = config.untouched_ratio
    training_sampler.calibration(training_loader, untouched_ratio=untouched_ratio, verbose=True)
    test_sampler.calibration(test_loader, untouched_ratio=untouched_ratio, verbose=True)

    # Optional debug functions
    # debug_timing(training_dataset, training_loader)
    # debug_timing(test_dataset, test_loader)
    # debug_upsampling(training_dataset, training_loader)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = KPFCNN(config, training_dataset.label_values, training_dataset.ignored_labels)

    debug = False
    if debug:
        print('\n*************************************\n')
        print(net)
        print('\n*************************************\n')
        for param in net.parameters():
            if param.requires_grad:
                print(param.shape)
        print('\n*************************************\n')
        print("Model size %i" % sum(param.numel() for param in net.parameters() if param.requires_grad))
        print('\n*************************************\n')

    # Define a trainer class & summary dir
    if tb_logdir is None:
        date_head = datetime.now().strftime("%Y-%m-%d")
        tb_logdir = Path(config.saving_path) / "logdir"
        cnt = len(list(tb_logdir.glob(date_head + "_*")))
        tb_logdir = tb_logdir / f"{date_head}_experiment_{cnt}"
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp, tb_logdir=tb_logdir, finetune=finetune)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    config.save()

    print('\nStart training')
    print('**************')

    # Gradual unfreezing, see unfreeze_schedule in utils/config_railtwin_tl.py
    trainer.train_with_unfreeze_schedule(net, training_loader, test_loader, config)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Launch from Docker')
    parser.add_argument('-o', '--results', type=Path, help="Path to wherever everything happens", default="~/workdir")
    parser.add_argument('-ptp', '--chkp_path', type=Path, help="Run folder or checkpoint to resume (weights, optimizer, epoch)", default="/default/thing/that/does/not/exist")
    parser.add_argument('-pt', '--pretrained', type=Path, help="Run folder or checkpoint of the pre-training (weights only, starts the unfreeze schedule at epoch 0)", default=None)
    parser.add_argument('-ft', "--fine_tune", action='store_true', default=False,
                        help="Deprecated: same as --pretrained <results>, i.e. the pre-trained checkpoint was copied into the output folder")
    parser.add_argument('-gid', "--gpu_id", type=str, default=None)
    # parser.add_argument('--config_path', type=Path, help="Config path to load", default="/default/thing/that/does/not/exist")
    args = parser.parse_args()
    # print(args)

    # args.chkp_path  = ""

    pretrained = args.pretrained.expanduser() if args.pretrained else None
    if args.fine_tune and pretrained is None:
        pretrained = args.results.expanduser()
    train_railtwin(args.results.expanduser(), args.chkp_path.expanduser(), pretrained_path=pretrained, gid=args.gpu_id)  # , args.config_path.expanduser())
