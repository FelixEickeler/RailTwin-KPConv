import os
from utils.new_config import Config
import numpy as np

class RailTwinConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'RailTwin'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializing dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = 'cloud_segmentation'

    # Number of CPU threads for the input pipeline
    input_threads = 12#int(np.floor(os.cpu_count() / 2))

    #########################
    # Architecture definition
    #########################

    # # Define layers
    # architecture = ['simple',
    #                 'resnetb',
    #                 'resnetb_strided',
    #                 'resnetb',
    #                 'resnetb',
    #                 'resnetb_strided',
    #                 'resnetb',
    #                 'resnetb',
    #                 'resnetb_strided',
    #                 'resnetb_deformable',
    #                 'resnetb_deformable',
    #                 'resnetb_deformable_strided',
    #                 'resnetb_deformable',
    #                 'resnetb_deformable',
    #                 'nearest_upsample',
    #                 'unary',
    #                 'nearest_upsample',
    #                 'unary',
    #                 'nearest_upsample',
    #                 'unary',
    #                 'nearest_upsample',
    #                 'unary']

    # Define layers: THIS IS S3DIS & NPM3D
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    # 'resnetb', # added
                    # 'resnetb', # added
                    # 'resnetb_strided', #added
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb',
                    'nearest_upsample',
                    'unary',
                    # 'nearest_upsample', #added
                    # 'unary', #added
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary']

    ###################
    # KPConv parameters
    ###################

    # Number of kernel points
    num_kernel_points = 15

    # Radius of the input sphere (decrease value to reduce memory cost)
    in_radius = 8.5

    # Size of the first subsampling grid in meter (increase value to reduce memory cost)
    first_subsampling_dl = 0.04

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 5.0

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    first_features_dim = 128
    in_features_dim = 2

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.02

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 350

    # Learning rate management
    learning_rate = 0.8e-2
    # learning_rate = 5e-3
    momentum = 0.98
    epoch_steps = 2000
    validation_size = 200
    plot_gap = 5
    checkpoint_gap = 25
    # TODO update lr_decay

    lr_decays = {i: 0.1 ** (1 / 50) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of batch (decrease to reduce memory cost, but it should remain > 3 for stability)
    batch_num = 3
    val_batch_num = 3
    # Number of steps per epochs


    # Number of validation examples per epoch



    # Number of test examples per epoch
    # test_size = 50

    # Number of epoch between each checkpoint


    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, True, False]
    augment_rotation = 'pendulum'
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_noise = 0.001
    augment_color = 0.8
    augment_height = 0.05

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = "results"

    #added
    untouched_ratio = 0.45


