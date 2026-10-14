#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Input features of the network, chosen by name in the config
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      RailTwin addition
#
#      config.input_features: list of feature names, e.g. ['above_ground'] or ['above_ground', 'intensity'] or []
#      (geometry only). The network input is [1, <input_features>], so config.in_features_dim must be
#      1 + len(input_features). Names are point fields of the clouds (e.g. 'above_ground', 'intensity') or
#      'height' (absolute z of the point).
#

import numpy as np

# Feature order of configs without input_features: in_features_dim d uses the first d - 1 of these
LEGACY_FEATURES = ['above_ground', 'intensity', 'height']


def feature_names(config):
    """Names of the input features (without the constant 1)."""
    names = config.input_features
    if names is None:
        names = LEGACY_FEATURES[:max(int(config.in_features_dim) - 1, 0)]
    names = list(names)
    if len(names) + 1 != int(config.in_features_dim):
        raise ValueError(f'in_features_dim must be 1 + len(input_features): input_features = {names} needs '
                         f'in_features_dim = {len(names) + 1}, not {config.in_features_dim}')
    return names


def select_features(columns, config, training=False):
    """
    Per-point input features (without the constant 1).

    :param columns: dict name -> float[N] of the available point fields, e.g. {'above_ground': ..., 'intensity': ...,
                    'height': ...}
    :param config: uses input_features (or in_features_dim), augment_color and drop_features
    :param training: feature dropout only for training samples: with probability 1 - augment_color all point fields
                     (not 'height') are zeroed, as in the original KPConv colour augmentation
    :return: float32[N, len(input_features)]
    """
    names = feature_names(config)
    n = len(next(iter(columns.values()))) if columns else 0
    missing = [name for name in names if name not in columns]
    if missing:
        raise KeyError(f'Input features {missing} not available, the cloud provides {sorted(columns)}')
    features = np.zeros((n, len(names)), dtype=np.float32)
    for i, name in enumerate(names):
        features[:, i] = columns[name]

    point_fields = [i for i, name in enumerate(names) if name != 'height']
    if training and np.random.rand() > config.augment_color:
        features[:, point_fields] = 0
    if config.drop_features:
        # Model trained with features, applied to data without them
        features[:, point_fields] = 0
    return features


def stack_features(stacked_points, features, config):
    """Network input features [1, features] of a batch, checked against in_features_dim."""
    stacked = np.ones_like(stacked_points[:, :1], dtype=np.float32)
    if features.shape[1] > 0:
        stacked = np.hstack((stacked, features.astype(np.float32)))
    if stacked.shape[1] != int(config.in_features_dim):
        raise ValueError(f'The batch has {stacked.shape[1]} input features, the network expects '
                         f'in_features_dim = {config.in_features_dim}')
    return stacked
