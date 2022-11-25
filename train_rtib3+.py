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

# Dataset
from torch.utils.data import DataLoader
from models.architectures import KPFCNN

# from custom_datasets.railtwin_ecppm2022 import *
# from datasets.S3DIS import S3DISDataset as RailTwinDatasetECPPM2022
# from datasets.RailTwin_Ecppm2022 import RailTwinDatasetECPPM2022
# from datasets.S3DIS import S3DISDataset as RailTwinDatasetECPPM2022
# from datasets.rtib3p import RTIB3p as RailTwinDatasetECPPM2022
from datasets.rtib3p_16classes import RTIB3p as RailTwinDatasetECPPM2022
from datasets.rtib3p import RTIB3pSampler as RailTwinSampler
from datasets.S3DIS import S3DISCollate as RailTwinCollate
from utils.trainer_railtwin import ModelTrainer
from utils.config_railtwin import RailTwinConfig


# from train_S3DIS import S3DISConfig as RailTwinConfig


def train_railtwin(working_path, previous_training_path=Path(''), fine_tune=False, gid=None):
    if fine_tune:
        # not sure about this
        previous_training_path = working_path

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

    ###############
    # Previous chkp
    ###############
    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chkp_idx = None
    if previous_training_path.exists() and (previous_training_path / 'checkpoints').exists():
        # Find all snapshot in the chosen training folder
        chkp_path = previous_training_path / 'checkpoints'
        if chkp_path.exists():
            chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

            # Find which snapshot to restore
            if chkp_idx is None:
                chosen_chkp = 'current_chkp.tar'
            else:
                chosen_chkp = np.sort(chkps)[chkp_idx]
            chosen_chkp = os.path.join('results', previous_training_path, 'checkpoints', chosen_chkp)
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

    # TODO Add new config
    if previous_training_path.exists() and (previous_training_path / 'parameters.txt').exists():
        config.load(str(previous_training_path))
        config.saving_path = None

    # Get path from argument if given
    if working_path.exists():
        config.saving_path = str(working_path)

    # Initialize datasets
    training_dataset = RailTwinDatasetECPPM2022(config, set='training', use_potentials=True)
    training_sampler = RailTwinSampler(training_dataset)

    test_dataset = RailTwinDatasetECPPM2022(config, set='validation', use_potentials=True)
    test_sampler = RailTwinSampler(test_dataset)

    # test_dataset = RailTwinDatasetECPPM2022(config, set='test', use_potentials=True)

    # Initialize samplers


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
    trainer = ModelTrainer(net, config, chkp_path=chosen_chkp, tb_logdir=tb_logdir, finetune=fine_tune)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    config.save()

    print('\nStart training')
    print('**************')

    # Training
    trainer.train(net, training_loader, test_loader, config)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Launch from Docker')
    parser.add_argument('-o', '--results', type=Path, help="Path to wherever everything happens", default="~/workdir")
    parser.add_argument('-ptp', '--chkp_path', type=Path, help="Checkpoint you want to continue from", default="/default/thing/that/does/not/exist")
    parser.add_argument('-ft', "--fine_tune", action='store_true', default=False)
    parser.add_argument('-gid', "--gpu_id", type=str, default=None)
    # parser.add_argument('--config_path', type=Path, help="Config path to load", default="/default/thing/that/does/not/exist")
    args = parser.parse_args()
    # print(args)

    # args.chkp_path  = ""

    train_railtwin(args.results.expanduser(), args.chkp_path.expanduser(), fine_tune=args.fine_tune, gid=args.gpu_id)  # , args.config_path.expanduser())
