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

from datasets.railtwin import RailTwinDataset, RailTwinSampler, RailTwinCollate
from utils.trainer_railtwin import ModelTrainer
from utils.config_railtwin_tl import RailTwinConfig
from utils.checkpoints import find_checkpoint


# from train_S3DIS import S3DISConfig as RailTwinConfig


def train_railtwin(working_path, previous_training_path=Path(''), pretrained_path=None, gid=None, overrides=None):
    """
    :param working_path: output folder of this run
    :param previous_training_path: run folder or checkpoint to resume (weights, optimizer and epoch are restored)
    :param pretrained_path: run folder or checkpoint of the pre-training. Only the weights are loaded, the fine-tuning
                            starts at epoch 0 and follows config.unfreeze_schedule.
    :param gid: CUDA_VISIBLE_DEVICES to use, None = pick free GPUs
    :param overrides: dict of config values to override (command line)
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

    # Command line overrides (e.g. --in-radius for sphere size experiments)
    for key, value in (overrides or {}).items():
        print(f"Config override: {key} = {value}")
        setattr(config, key, value)

    # Initialize datasets
    # Class-balanced training spheres (config.balance_classes) instead of potentials; validation keeps potentials
    training_dataset = RailTwinDataset(config, set='training',
                                                use_potentials=not config.balance_classes)
    test_dataset = RailTwinDataset(config, set='validation', use_potentials=True)
    # test_dataset = RailTwinDataset(config, set='test', use_potentials=True)

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
    parser.add_argument('-gid', "--gpu_id", type=str, default=None)
    parser.add_argument('-r', '--in-radius', type=float, default=None,
                        help="Override config.in_radius [m] (input sphere radius), e.g. for 2.5 / 4 / 6 m experiments")
    parser.add_argument('--schedule', choices=['gradual', 'real'], default='gradual',
                        help="Unfreeze schedule: 'gradual' (unfreeze_schedule, layer by layer) or 'real' "
                             "(unfreeze_schedule_real: short head warm-up, then the whole network at reduced LR)")
    # parser.add_argument('--config_path', type=Path, help="Config path to load", default="/default/thing/that/does/not/exist")
    args = parser.parse_args()
    # print(args)

    # args.chkp_path  = ""

    pretrained = args.pretrained.expanduser() if args.pretrained else None
    overrides = {'in_radius': args.in_radius} if args.in_radius is not None else {}
    if args.schedule == 'real':
        overrides['unfreeze_schedule'] = RailTwinConfig.unfreeze_schedule_real
    train_railtwin(args.results.expanduser(), args.chkp_path.expanduser(), pretrained_path=pretrained, gid=args.gpu_id,
                   overrides=overrides)  # , args.config_path.expanduser())
