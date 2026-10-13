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

    # Choice of input features (datasets/features.py): network input = [1, input_features], so
    # in_features_dim = 1 + len(input_features). Names: 'above_ground', 'intensity', 'height' (absolute z).
    # [] = geometry only.
    first_features_dim = 128
    input_features = ['above_ground']
    in_features_dim = 2
    drop_features = False                   # zero the input features of every point (data without them)

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

    # Gradual unfreezing for transfer learning (used with --pretrained).
    # Each stage makes one more part of the network trainable (on top of the previous stages)
    # and trains it for the given number of epochs, counted from the start of the fine-tuning.
    # None = train until max_epoch. An empty list trains the whole network from the start.
    # Parts: 'head_softmax', 'head_mlp', 'decoder_blocks', 'encoder_blocks'
    unfreeze_schedule = [
        ['head_softmax', 11],
        ['head_mlp', 20],
        ['decoder_blocks', 20],
        ['encoder_blocks', None],
    ]

    # Fine-tuning on real data (train_rtib3+_tl.py --schedule real): a short warm-up of the head, then the whole
    # network at a reduced learning rate. Literature: long freezing costs accuracy on real data, full fine-tuning
    # after a short warm-up is the usual recipe. A stage may unfreeze several parts at once; the optional third value
    # is a learning rate factor (relative to learning_rate) from that stage on.
    unfreeze_schedule_real = [
        ['head_softmax', 2],
        [['head_mlp', 'decoder_blocks', 'encoder_blocks'], None, 0.1],
    ]

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
    # 'pendulum': random heading + swing of at most augment_rotation_max_tilt degrees (datasets/rotations.py).
    # 'pendulum_legacy' reproduces the original RailTwin rotation (often upside down) of older trainings.
    augment_rotation = 'pendulum'
    augment_rotation_max_tilt = 10.0
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_noise = 0.001
    augment_color = 0.8
    augment_height = 0.05

    # ALS augmentation (utils/als_simulation.py): mixes ALS-like samples into the training.
    # augment_als = share of training spheres taken from an ALS emulation (0 = off).
    # 'prepared': at dataset load, augment_als_versions ALS versions of every training cloud are computed once
    #             (flight strips over the whole cloud, headings spread over 180 deg) and cached in work_dir.
    # 'online':   each sphere is emulated on the fly from one of augment_als_planes (only the sphere occludes).
    augment_als = 0.0
    augment_als_mode = 'prepared'
    augment_als_versions = 2
    augment_als_height = 100.0              # [m] flying height above the lowest point
    augment_als_max_scan_angle = 30.0       # [deg] half swath angle
    augment_als_side_overlap = 0.3          # overlap of neighbouring flight strips
    augment_als_footprint = 0.3             # [m] beam footprint
    augment_als_returns = 1                 # returns per beam
    augment_als_voxel = None                # [m] voxel mode
    augment_als_planes = [                  # 'online' only: height [m], off_nadir [deg] or offset [m]
        {'height': 100.0, 'off_nadir': 0.0},
        {'height': 100.0, 'off_nadir': 15.0},
        {'height': 100.0, 'off_nadir': 25.0},
    ]
    augment_als_random_heading = True       # 'online' only

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'                # deprecated in KPConv, has no effect (use class_w / segmentation_loss)

    # Segmentation loss (models/losses.py): weights of cross entropy and Lovasz-softmax (optimises the IoU directly,
    # commonly combined 1:1 for LiDAR segmentation), label smoothing of the cross entropy (0 = off, 0.1-0.2 usual)
    segmentation_loss = {'cross_entropy': 1.0, 'lovasz': 1.0}
    label_smoothing = 0.0

    # Class-balanced training spheres: centres are drawn per class instead of by potentials (training set only,
    # validation keeps potentials). Recommended by the KPConv author for rare classes.
    balance_classes = True

    # Do we nee to save convergence
    saving = True
    saving_path = "results"

    #added
    untouched_ratio = 0.45


