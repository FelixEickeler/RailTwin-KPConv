# 21.06.2022----------------------------------------------------------------------------------------------------------------------
#  created by: felix
#              felix@eickeler.com
# ----------------------------------------------------------------------------------------------------------------
#
import numpy as np
from pathlib import Path
from os.path import join
import json
from collections import OrderedDict
# Colors for printing
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Config:
    dataset = ''                    # Dataset name
    dataset_task = ''               # Type of network model
    num_classes = 0                 # Number of classes in the dataset
    features = ["x", "y", "z"]      # features to be taken
    in_points_dim = 3               # Dimension of input points
    in_features_dim = 2             # Dimension of input features
    in_radius = 8.9                 # Radius of the input sphere (ignored for models, only used for point clouds)
    input_threads = 12               # Number of CPU threads for the input pipeline
    architecture = []               # Architecture definition. List of blocks
    equivar_mode = ''               # Decide the mode of equivariance and invariance
    invar_mode = ''                 # Decide the mode of equivariance and invariance
    first_features_dim = 128         # Dimension of the first feature maps
    use_batch_norm = True           # Batch normalization parameters
    batch_norm_momentum = 0.99      # Batch normalization parameters
    segmentation_ratio = 1.0        # For segmentation models : ratio between the segmented area and the input area
    num_kernel_points = 15          # Number of kernel points
    first_subsampling_dl = 0.02     # Size of the first subsampling grid in meter
    conv_radius = 2.5               # Radius of convolution in "number grid cell". (2.5 is the standard value)
    deform_radius = 5.0             # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    KP_extent = 1.0                 # Kernel point influence radius
    KP_influence = 'linear'         # Influence function when d < KP_extent. ('constant', 'linear', 'gaussian') When d > KP_extent, always zero
    aggregation_mode = 'sum'        # Aggregation function of KPConv in ('closest', 'sum') Decide if you sum all kernel point influences, or if you only take the influence of the closest KP
    fixed_kernel_points = 'center'  # Fixed points in the kernel : 'none', 'center' or 'verticals'
    modulated = False               # Use modulateion in deformable convolutions
    n_frames = 1                    # For SLAM datasets like SemanticKitti number of frames used (minimum one)
    max_in_points = 0               # For SLAM datasets like SemanticKitti max number of point in input cloud + validation
    val_radius = 51.0               # For SLAM datasets like SemanticKitti max number of point in input cloud + validation
    max_val_points = 50000          # For SLAM datasets like SemanticKitti max number of point in input cloud + validation

    # Training parameters
    learning_rate = 1e-3            # Network optimizer parameters (learning rate and momentum)
    momentum = 0.9                  # Network optimizer parameters (learning rate and momentum)
    lr_decays = {200: 0.2, 300: 0.2}    # Learning rate decays. Dictionary of all decay values with their epoch {epoch: decay}.
    grad_clip_norm = 100.0          # Gradient clipping value (negative means no clipping)

    augment_scale_anisotropic = True            # Augmentation parameters
    augment_scale_min = 0.9                     # Augmentation parameters
    augment_scale_max = 1.1                     # Augmentation parameters
    augment_symmetries = [False, False, False]  # Augmentation parameters
    augment_rotation = 'vertical'               # Augmentation parameters
    augment_noise = 0.005                       # Augmentation parameters
    augment_color = 0.7                         # Augmentation parameters
    augment_occlusion = 'none'                  # Augment with occlusions (not implemented yet)
    augment_occlusion_ratio = 0.2               # Augment with occlusions (not implemented yet)
    augment_occlusion_num = 1                   # Augment with occlusions (not implemented yet)

    weight_decay = 1e-3                         # Regularization loss importance
    class_w = []                                # Choose weights for class (used in segmentation loss). Empty list for no weights

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.0                    # Distance of repulsion for deformed kernel points
    batch_num = 10                          # Number of batch
    val_batch_num = 10                      # Number val batch
    max_epoch = 1000                        # Maximal number of epochs
    epoch_steps = 1000                      # Number of steps per epochs
    validation_size = 100                   # Number of validation examples per epoch
    checkpoint_gap = 50                     # Number of epoch between each checkpoint
    saving = True                           # Do we nee to save convergence
    saving_path = None

    version = 2.0
    pot_sample_mult: 50
    tukey_update_crop: 0.7
    untouched_ratio = 0.9





    def __init__(self):
        # Number of layers
        self.num_layers = len([block for block in self.architecture if 'pool' in block or 'strided' in block]) + 1

        ###################
        # Deform layer list
        ###################
        #
        # List of boolean indicating which layer has a deformable convolution
        #

        layer_blocks = []
        self.deform_layers = []
        arch = self.architecture
        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            # *****************************
            deform_layer = False
            if layer_blocks:
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    deform_layer = True

            if 'pool' in block or 'strided' in block:
                if 'deformable' in block:
                    deform_layer = True

            self.deform_layers += [deform_layer]
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.is_dir():
            path =  path / 'parameters.txt'

        if path.suffix == ".txt":
            this = cls.load_txt(str(path))
        else:
            this = cls.load_json(path)
        return this

    @classmethod
    def load_json(cls, path):
        with open(path, "r") as file:
            # This is used to initalize the class from parsed_configs
            data = json.load(file)

        for key, value in data.items():
            setattr(cls, key, value)
        # cls.__dict__.update(data)
        return cls()

    @classmethod
    def load_txt(cls, path):
        with open(path, 'r') as f:
            lines = f.readlines()

        # Class variable dictionary
        for line in lines:
            line_info = line.split()
            if len(line_info) > 2 and line_info[0] != '#':

                if line_info[2] == 'None':
                    setattr(cls, line_info[0], None)

                elif line_info[0] == 'lr_decay_epochs':
                    cls.lr_decays = {int(b.split(':')[0]): float(b.split(':')[1]) for b in line_info[2:]}

                elif line_info[0] == 'architecture':
                    cls.architecture = [b for b in line_info[2:]]

                elif line_info[0] == 'augment_symmetries':
                    cls.augment_symmetries = [bool(int(b)) for b in line_info[2:]]

                elif line_info[0] == 'num_classes':
                    if len(line_info) > 3:
                        cls.num_classes = [int(c) for c in line_info[2:]]
                    else:
                        cls.num_classes = int(line_info[2])

                elif line_info[0] == 'class_w':
                    cls.class_w = [float(w) for w in line_info[2:]]

                elif hasattr(cls, line_info[0]):
                    attr_type = type(getattr(cls, line_info[0]))
                    if attr_type == bool:
                        setattr(cls, line_info[0], attr_type(int(line_info[2])))
                    else:
                        setattr(cls, line_info[0], attr_type(line_info[2]))

        cls.saving = True
        cls.saving_path = path
        return cls()

    def copy(self):
        cls = self.__class__()
        data = {key:value for key, value in self.__dict__.items()}
        for key, value in data.items():
            setattr(cls, key, value)
        return cls



    def save(self):
        target = Path(self.saving_path)
        if not target.is_dir():
            target = target.parent
        with open(target / 'parameters.json', "w") as file:
            dat = OrderedDict({key:value for key, value in self.__class__.__dict__.items()})
            dat.move_to_end("lr_decays", last=True)
            json.dump(dat, file, indent=4)

    @property
    def label_selector(self):
        return self.pred_names.index(self.pred_selected)
