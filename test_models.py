#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on ModelNet40 dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import signal
import os
from pathlib import Path

import numpy as np
import sys
import torch

# Dataset
from datasets.ModelNet40 import *
from datasets.S3DIS import *
from datasets.SemanticKitti import *
from datasets.rtib3p import RTIB3pSampler
import importlib
from torch.utils.data import DataLoader


from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPCNN, KPFCNN
import argparse


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def model_choice(chosen_log):
    ###########################
    # Call the test initializer
    ###########################

    # Automatically retrieve the last trained model
    if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS', 'last_rtib+']:

        # Dataset name
        test_dataset = '_'.join(chosen_log.split('_')[1:])

        # List all training logs
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])

        # Find the last log of asked dataset
        for log in logs[::-1]:
            log_config = Config()
            log_config.load(log)
            if log_config.dataset.startswith(test_dataset):
                chosen_log = log
                break

        if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    return chosen_log


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Model testing")
    parser.add_argument('-lp', '--log_dir', required=True, type=Path,
                        help='Path to log_dir, used for testing')
    parser.add_argument('-mp', '--model_path', type=Path, default=None,
                        help='Path to the model, used for testing')
    parser.add_argument('-o', '--out_dir', default=None, type=Path,
                        help="Output directory: default is _tested same level as checkpoints")
    parser.add_argument('-v', '--val', default="test", type=str,
                        help="Choose to test on validation or test split")
    parser.add_argument('-gid', '--gpu_id', default="0", type=str,
                        help="Choose the GPU_ID for training")
    parser.add_argument('-do', '--dataset_overwrite', default="rtib3p", type=str,
                        help="Name the file of the dataset with an RTIB+ class")
    args = parser.parse_args()

    ###############################
    # Choose the model to visualize
    ###############################

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

    log_dir = args.log_dir
    model_path = args.model_path
    output_path = args.out_dir
    # validation_split = args.val == "test"
    dataset_overwrite = args.dataset_overwrite

    if output_path is None:
        output_path = log_dir / args.val
    output_path.mkdir(exist_ok=True, parents=True)

    if model_path is None:
        model_path = log_dir / "checkpoints"

    if model_path.is_dir():
        chkps = list(model_path.glob("chkp_*.tar"))
        chkps.sort(key=lambda x: os.path.getmtime(x))
        model_path = chkps[-1].resolve()


    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = args.gpu_id

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    # Initialize configuration class
    config = Config()
    config.load(str(log_dir))

    if dataset_overwrite != config.dataset and dataset_overwrite != "rtib+":
        dataset_import = importlib.import_module(f'datasets.{dataset_overwrite}')
        RTIB3p = dataset_import.RTIB3p
        # RTIB3p = getattr(__import__(f"datasets.{dataset_overwrite}"), "*")

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    # config.augment_noise = 0.0001
    # config.augment_symmetries = False

    config.augment_scale_anisotropic = False
    config.augment_symmetries = [False, False, False]
    # config.batch_num = 3
    # config.in_radius = 4
    config.validation_size = 200
    config.input_threads = 20

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initiate dataset
    if config.dataset == 'ModelNet40':
        test_dataset = ModelNet40Dataset(config, train=False)
        test_sampler = ModelNet40Sampler(test_dataset)
        collate_fn = ModelNet40Collate
    elif config.dataset == 'S3DIS':
        test_dataset = S3DISDataset(config, set='validation', use_potentials=True)
        test_sampler = S3DISSampler(test_dataset)
        collate_fn = S3DISCollate
    elif config.dataset == 'SemanticKitti':
        test_dataset = SemanticKittiDataset(config, set=set, balance_classes=False)
        test_sampler = SemanticKittiSampler(test_dataset)
        collate_fn = SemanticKittiCollate
    elif config.dataset in ['rtib+', "RailTwin", "-rtib"]:
        test_dataset = RTIB3p(config, set='validation', use_potentials=True)
        test_sampler = RTIB3pSampler(test_dataset)
        collate_fn = S3DISCollate
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    os.chdir(output_path)

    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    if config.dataset_task == 'classification':
        net = KPCNN(config)
    elif config.dataset_task in ['cloud_segmentation', 'slam_segmentation']:
        net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=model_path)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')

    # Training
    if config.dataset_task == 'classification':
        tester.classification_test(net, test_loader, config)
    elif config.dataset_task == 'cloud_segmentation':
        tester.cloud_segmentation_test(net, test_loader, config)
    elif config.dataset_task == 'slam_segmentation':
        tester.slam_segmentation_test(net, test_loader, config)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)
