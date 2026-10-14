"""
Tests for the input feature selection (datasets/features.py).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from datasets.features import feature_names, select_features, stack_features

N = 1000


def columns():
    rng = np.random.default_rng(0)
    return {'above_ground': rng.uniform(0, 5, N), 'intensity': rng.uniform(0, 1, N), 'height': rng.uniform(500, 520, N)}


def cfg(**kwargs):
    defaults = dict(in_features_dim=2, augment_color=0.8, input_features=None, drop_features=False)
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def legacy_potential_features(colors, in_features_dim, augment_color, training=True):
    """The feature code of rtib3p potential_item before the change (dims 1-3, the valid ones)."""
    if np.random.rand() > augment_color:
        colors = colors * 0
    features = np.hstack((colors[:, :1], colors[:, 1:2])).astype(np.float32)
    stacked = np.ones((len(colors), 1), dtype=np.float32)
    if in_features_dim == 2:
        stacked = np.hstack((stacked, features[:, :1]))
    elif in_features_dim == 3:
        stacked = np.hstack((stacked, features[:, :2]))
    return stacked


@pytest.mark.parametrize('dim, names', [(1, []), (2, ['above_ground']), (3, ['above_ground', 'intensity']),
                                        (4, ['above_ground', 'intensity', 'height'])])
def test_legacy_configs_keep_their_features(dim, names):
    assert feature_names(cfg(in_features_dim=dim)) == names


@pytest.mark.parametrize('dim', [1, 2, 3])
def test_training_features_identical_to_the_old_code(dim):
    c = columns()
    colors = np.stack([c['above_ground'], c['intensity']], axis=1)
    config = cfg(in_features_dim=dim)
    for seed in range(30):
        np.random.seed(seed)
        expected = legacy_potential_features(colors, dim, config.augment_color)
        expected_next = np.random.rand()
        np.random.seed(seed)
        actual = stack_features(np.zeros((N, 3)), select_features(c, config, training=True), config)
        actual_next = np.random.rand()
        np.testing.assert_array_equal(actual, expected)
        assert actual_next == expected_next


def test_named_features():
    c = columns()
    config = cfg(input_features=['intensity', 'height'], in_features_dim=3, augment_color=1.0)
    f = select_features(c, config, training=True)
    np.testing.assert_allclose(f, np.stack([c['intensity'], c['height']], axis=1).astype(np.float32))
    assert stack_features(np.zeros((N, 3)), f, config).shape == (N, 3)


def test_geometry_only():
    config = cfg(input_features=[], in_features_dim=1)
    f = select_features(columns(), config, training=True)
    assert f.shape == (N, 0)
    np.testing.assert_array_equal(stack_features(np.zeros((N, 3)), f, config), np.ones((N, 1)))


def test_dim_mismatch_is_reported():
    with pytest.raises(ValueError, match='in_features_dim'):
        feature_names(cfg(input_features=['above_ground', 'intensity'], in_features_dim=2))
    with pytest.raises(ValueError, match='in_features_dim'):
        stack_features(np.zeros((N, 3)), np.zeros((N, 2)), cfg(in_features_dim=2))


def test_missing_feature_is_reported():
    with pytest.raises(KeyError, match='colour'):
        select_features(columns(), cfg(input_features=['colour'], in_features_dim=2))


def test_dropout_only_for_training():
    c = columns()
    config = cfg(augment_color=0.0)  # training: always dropped
    np.random.seed(0)
    assert np.all(select_features(c, config, training=True) == 0)
    # Validation / prediction: never dropped, and no random number drawn
    np.random.seed(0)
    f = select_features(c, config, training=False)
    np.testing.assert_allclose(f[:, 0], c['above_ground'].astype(np.float32))
    assert np.random.rand() == np.random.RandomState(0).rand()


def test_dropout_keeps_height():
    c = columns()
    config = cfg(input_features=['above_ground', 'height'], in_features_dim=3, augment_color=0.0)
    np.random.seed(0)
    f = select_features(c, config, training=True)
    assert np.all(f[:, 0] == 0)
    np.testing.assert_allclose(f[:, 1], c['height'].astype(np.float32))


def test_drop_features_for_prediction():
    c = columns()
    config = cfg(input_features=['above_ground', 'intensity'], in_features_dim=3, drop_features=True)
    f = select_features(c, config, training=False)
    assert f.shape == (N, 2) and np.all(f == 0)
