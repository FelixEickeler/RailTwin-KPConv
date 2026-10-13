"""
Tests for the rotation augmentation (datasets/rotations.py).

'pendulum_legacy' is the original RailTwin 'pendulum'. These tests pin down what it actually does, so that old
trainings stay reproducible. 'pendulum' is the corrected "random heading + bounded swing" mode.
"""
import numpy as np
import pytest

from datasets.rotations import random_rotation, PENDULUM_DEFAULT_YAW
from kernels.kernel_points import create_3D_rotations

N_SAMPLES = 2000
Z = np.array([0.0, 0.0, 1.0])


def sample(mode, n=N_SAMPLES, seed=0, **kwargs):
    np.random.seed(seed)
    return np.stack([random_rotation(mode, **kwargs) for _ in range(n)])


def tilt_deg(Rs):
    """Angle between the rotated z-axis and the vertical, in degrees."""
    return np.rad2deg(np.arccos(np.clip(Rs[:, 2, 2], -1, 1)))


def legacy_augmentation_rotation(augment_rotation, yaw_config=None):
    """Verbatim copy of the rotation block of PointCloudDataset.augmentation_transform before the refactoring."""
    R = np.eye(3)
    if augment_rotation == 'vertical':
        theta = np.random.rand() * 2 * np.pi
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    elif augment_rotation == 'all' or augment_rotation == "pendulum":
        if augment_rotation == "pendulum":
            try:
                yaw = yaw_config / 360
            except:
                yaw = 0.028
        else:
            yaw = 1
        theta = np.random.rand() * 2 * np.pi
        phi = (np.random.rand() - 0.5) * np.pi * yaw
        u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])
        alpha = np.random.rand() * 2 * np.pi
        R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]
    return R.astype(np.float32)


@pytest.mark.parametrize('mode', ['none', 'vertical', 'all', 'pendulum', 'pendulum_legacy'])
def test_is_proper_rotation(mode):
    Rs = sample(mode, n=200).astype(np.float64)
    eye = np.broadcast_to(np.eye(3), Rs.shape)
    np.testing.assert_allclose(Rs @ Rs.transpose(0, 2, 1), eye, atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(Rs), 1.0, atol=1e-5)


@pytest.mark.parametrize('legacy_mode, mode, yaw_config', [('vertical', 'vertical', None), ('all', 'all', None),
                                                           ('pendulum', 'pendulum_legacy', None),
                                                           ('pendulum', 'pendulum_legacy', 10.0)])
def test_legacy_modes_are_bit_identical(legacy_mode, mode, yaw_config):
    """Same seed -> same matrices as the original code (old 'pendulum' = 'pendulum_legacy')."""
    pendulum_yaw = None if yaw_config is None else yaw_config / 360
    for seed in range(20):
        np.random.seed(seed)
        expected = legacy_augmentation_rotation(legacy_mode, yaw_config)
        expected_next = np.random.rand()
        np.random.seed(seed)
        actual = random_rotation(mode, pendulum_yaw=pendulum_yaw)
        actual_next = np.random.rand()
        np.testing.assert_array_equal(actual, expected)
        # Same number of random draws, so the rest of the augmentation sees the same random stream
        assert actual_next == expected_next


def test_vertical_keeps_z_axis():
    Rs = sample('vertical')
    np.testing.assert_allclose(Rs @ Z, np.broadcast_to(Z, (len(Rs), 3)), atol=1e-6)


# ----------------------------------------------------------------------------------------------------------------------
# 'pendulum_legacy': characterisation of the original behaviour
# ----------------------------------------------------------------------------------------------------------------------

def test_pendulum_axis_elevation_is_a_quarter_of_the_configured_degrees():
    """phi = (rand - 0.5) * pi * deg / 360  ->  |phi| <= deg / 4. The default 0.028 ('10 deg') gives about 2.5 deg."""
    np.random.seed(0)
    max_phi = 0.0
    for _ in range(N_SAMPLES):
        np.random.rand()  # theta
        phi = (np.random.rand() - 0.5) * np.pi * PENDULUM_DEFAULT_YAW
        np.random.rand()  # alpha
        max_phi = max(max_phi, abs(np.rad2deg(phi)))
    assert max_phi <= 0.028 * 90 + 1e-9
    assert max_phi == pytest.approx(2.52, abs=0.05)


def test_pendulum_rotates_far_beyond_a_small_tilt():
    """Documents the known issue: the rotation angle is uniform in [0, 2pi) around an almost horizontal axis,
    so the cloud is often tilted by more than 90 deg (upside down)."""
    tilts = tilt_deg(sample('pendulum_legacy'))
    assert np.mean(tilts > 90) > 0.4
    assert tilts.max() > 170


# ----------------------------------------------------------------------------------------------------------------------
# 'pendulum': random heading + bounded swing
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('max_tilt', [2.0, 10.0, 30.0])
def test_pendulum_is_bounded(max_tilt):
    tilts = tilt_deg(sample('pendulum', max_tilt=max_tilt))
    assert tilts.max() <= max_tilt + 1e-3
    # The bound is actually used, not just a fraction of it
    assert tilts.max() > 0.95 * max_tilt


def test_pendulum_zero_equals_vertical_rotation():
    Rs = sample('pendulum', max_tilt=0.0, n=200)
    np.testing.assert_allclose(Rs @ Z, np.broadcast_to(Z, (len(Rs), 3)), atol=1e-6)


def test_pendulum_heading_covers_full_circle():
    Rs = sample('pendulum', max_tilt=10.0)
    x_rot = Rs @ np.array([1.0, 0.0, 0.0])
    heading = np.degrees(np.arctan2(x_rot[:, 1], x_rot[:, 0]))
    hist, _ = np.histogram(heading, bins=12, range=(-180, 180))
    # Roughly uniform heading: every 30 deg sector gets a share
    assert hist.min() > 0.5 * N_SAMPLES / 12


def test_pendulum_direction_is_isotropic():
    Rs = sample('pendulum', max_tilt=10.0)
    z_rot = Rs @ Z
    direction = np.degrees(np.arctan2(z_rot[:, 1], z_rot[:, 0]))
    hist, _ = np.histogram(direction, bins=8, range=(-180, 180))
    assert hist.min() > 0.5 * N_SAMPLES / 8
