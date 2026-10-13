#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Mix3D augmentation for training batches
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      RailTwin addition
#
#      Mix3D (Nekrasov et al., 3DV 2021, "Mix3D: Out-of-Context Data Augmentation for 3D Scenes"): two training
#      samples are overlaid into one, so the network has to segment objects out of their usual context. The authors
#      measured +0.7 to +1.5 mIoU with KPConv itself (mixing input spheres in the official KPConv code).
#
#      In KPConv a batch is a list of spheres, each centred at the origin and augmented on its own. With probability
#      config.augment_mix3d, two consecutive spheres of a batch are merged into one sample (points, features and
#      labels concatenated). The number of points per batch does not change.
#

import numpy as np


def mix_spheres(lists, prob):
    """
    Merge pairs of consecutive spheres of a batch.

    :param lists: dict name -> list with one entry per sphere. 'points', 'features', 'labels' and 'input_inds' are
                  concatenated when merged, the others ('point_inds', 'cloud_inds', 'scales', 'rots') keep the
                  value of the first sphere of the pair.
    :param prob: probability that a sphere is merged with the next one (0 = off, no random number is drawn)
    :return: dict with the same keys and the merged lists
    """
    n = len(lists['points'])
    if prob <= 0 or n < 2:
        return lists
    concatenated = {'points', 'features', 'labels', 'input_inds'}
    merged = {key: [] for key in lists}
    i = 0
    while i < n:
        if i + 1 < n and np.random.rand() < prob:
            for key, values in lists.items():
                if key in concatenated:
                    merged[key].append(np.concatenate((values[i], values[i + 1]), axis=0))
                else:
                    merged[key].append(values[i])
            i += 2
        else:
            for key, values in lists.items():
                merged[key].append(values[i])
            i += 1
    return merged
