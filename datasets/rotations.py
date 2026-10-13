#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Random rotations used for data augmentation (see PointCloudDataset.augmentation_transform)
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      RailTwin addition
#

import numpy as np

from kernels.kernel_points import create_3D_rotations

# Default of the 'pendulum_legacy' mode when config.augment_rotation_yaw is not set
PENDULUM_DEFAULT_YAW = 0.028


def random_rotation(mode, pendulum_yaw=None, max_tilt=10.0):
    """
    Draw a random 3x3 rotation matrix.

    :param mode: 'none', 'vertical', 'all', 'pendulum' or 'pendulum_legacy'
        - 'vertical': random rotation around the z-axis.
        - 'all': uniformly random 3D rotation.
        - 'pendulum': random rotation around the z-axis plus a swing of at most max_tilt degrees around a random
          horizontal axis, like a pendulum swinging below its pivot. The z-axis stays within max_tilt degrees of
          vertical.
        - 'pendulum_legacy': the 'pendulum' of the original RailTwin code, kept bit-identical to reproduce old
          trainings. It rotates by a random angle in [0, 2pi) around an axis whose elevation is limited to
          +-pendulum_yaw * 90 deg, i.e. around an almost horizontal axis, which often turns the cloud upside down.
    :param pendulum_yaw: 'pendulum_legacy' only, config.augment_rotation_yaw / 360 (None = PENDULUM_DEFAULT_YAW)
    :param max_tilt: 'pendulum' only, swing amplitude in degrees
    :return: float32[3, 3]
    """

    if mode == 'vertical':
        theta = np.random.rand() * 2 * np.pi
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

    elif mode in ('all', 'pendulum_legacy'):
        if mode == 'pendulum_legacy':
            yaw = PENDULUM_DEFAULT_YAW if pendulum_yaw is None else pendulum_yaw
        else:
            yaw = 1

        # Choose two random angles for the first vector in polar coordinates
        theta = np.random.rand() * 2 * np.pi
        phi = (np.random.rand() - 0.5) * np.pi * yaw

        # Create the first vector in carthesian coordinates
        u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

        # Choose a random rotation angle
        alpha = np.random.rand() * 2 * np.pi

        # Create the rotation matrix with this vector and angle
        R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

    elif mode == 'pendulum':
        # Random heading
        theta = np.random.rand() * 2 * np.pi
        c, s = np.cos(theta), np.sin(theta)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # Swing around a random horizontal axis
        beta = np.random.rand() * 2 * np.pi
        axis = np.array([[np.cos(beta), np.sin(beta), 0.0]])
        tilt = (np.random.rand() * 2 - 1) * np.deg2rad(max_tilt)
        R = create_3D_rotations(axis, np.reshape(tilt, (1, -1)))[0] @ Rz

    else:
        R = np.eye(3)

    return R.astype(np.float32)
