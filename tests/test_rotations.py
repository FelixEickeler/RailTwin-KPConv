"""
Tests for the rotation augmentation (datasets/rotations.py): 'pendulum' = random heading + bounded swing.
"""
import numpy as np
import pytest

from datasets.rotations import random_rotation

N_SAMPLES = 2000
Z = np.array([0.0, 0.0, 1.0])


def sample(mode, n=N_SAMPLES, seed=0, **kwargs):
    np.random.seed(seed)
    return np.stack([random_rotation(mode, **kwargs) for _ in range(n)])


def tilt_deg(Rs):
    """Angle between the rotated z-axis and the vertical, in degrees."""
    return np.rad2deg(np.arccos(np.clip(Rs[:, 2, 2], -1, 1)))


@pytest.mark.parametrize('mode', ['none', 'vertical', 'all', 'pendulum'])
def test_is_proper_rotation(mode):
    Rs = sample(mode, n=200).astype(np.float64)
    eye = np.broadcast_to(np.eye(3), Rs.shape)
    np.testing.assert_allclose(Rs @ Rs.transpose(0, 2, 1), eye, atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(Rs), 1.0, atol=1e-5)


def test_vertical_keeps_z_axis():
    Rs = sample('vertical')
    np.testing.assert_allclose(Rs @ Z, np.broadcast_to(Z, (len(Rs), 3)), atol=1e-6)


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


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        random_rotation('pendulum_legacy')
