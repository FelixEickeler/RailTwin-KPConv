"""
Tests for the ALS training augmentation: prepared versions and mixing (utils/als_simulation.py) and the dataset
methods (PointCloudDataset.prepare_als / als_augmentation in datasets/common.py).
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from utils.als_simulation import ALSConfig, emulate_als
from test_als_simulation import box, ground

GEO = np.array([691000.0, 5334000.0, 520.0])


def make_config(**kwargs):
    defaults = dict(augment_als=1.0, augment_als_versions=2, augment_als_height=100.0, augment_als_max_scan_angle=30.0,
                    augment_als_side_overlap=0.3, augment_als_footprint=0.3, augment_als_returns=1,
                    augment_als_min_points=2)
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.fixture(scope='module')
def sphere():
    """Input sphere (radius 9 m) of a georeferenced scene with a building at the centre."""
    roof, facades = box()
    parts = [('ground', ground()), ('roof', roof)] + [(f'facade{k}', v) for k, v in facades.items()]
    points = np.concatenate([p for _, p in parts])
    labels = np.concatenate([[name] * len(p) for name, p in parts])
    inside = np.linalg.norm(points - np.array([0.0, 0.0, 3.0]), axis=1) < 9.0
    center = np.array([0.0, 0.0, 3.0]) + GEO
    return points[inside] + GEO, labels[inside], center


# ----------------------------------------------------------------------------------------------------------------------
# Dataset methods (datasets/common.py), loaded without importing torch / the C++ wrappers
# ----------------------------------------------------------------------------------------------------------------------

import time as _time
import utils.als_simulation as als_module


def load_method(name, **namespace):
    """Function object of PointCloudDataset.<name>, executed with the given module namespace."""
    source = (Path(__file__).resolve().parents[1] / 'datasets' / 'common.py').read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'PointCloudDataset')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.name = '_method'  # as a method it does not shadow module functions of the same name
    defaults = dict(np=np, time=_time,
                    prepare_als_versions=als_module.prepare_als_versions,
                    prepared_als_mask=als_module.prepared_als_mask,
                    prepared_als_settings=als_module.prepared_als_settings)
    defaults.update(namespace)
    exec(compile(ast.Module([fn], []), 'common.py', 'exec'), defaults)
    return defaults['_method']


class FakeTree:
    def __init__(self, data):
        self.data = data


def fake_dataset(tmp_path, clouds, split='training', **config):
    cfg = make_config(**{'first_subsampling_dl': 0.04, **config})
    return SimpleNamespace(set=split, config=cfg, work_dir=str(tmp_path), cloud_names=list(clouds), label_mapping=None,
                           input_trees=[FakeTree(p) for p in clouds.values()])


@pytest.mark.parametrize('split', ['validation', 'test', 'ERF'])
def test_hook_only_for_training(sphere, split):
    points, _, _ = sphere
    hook = load_method('als_augmentation')
    versions = [np.packbits(np.ones((2, len(points)), dtype=bool), axis=1)]
    ds = SimpleNamespace(set=split, config=make_config(), als_versions=versions)
    assert hook(ds, 0, np.arange(len(points))) is None


# ----------------------------------------------------------------------------------------------------------------------
# Prepared versions: preparation step + mixing
# ----------------------------------------------------------------------------------------------------------------------

SETTINGS = dict(versions=3, height=100.0, max_scan_angle=30.0, side_overlap=0.3, footprint=0.3, returns=1, algorithm=1)


def test_prepare_versions_shape_and_determinism(sphere):
    points, _, _ = sphere
    packed = als_module.prepare_als_versions(points, SETTINGS, seed=7)
    assert packed.shape == (3, int(np.ceil(len(points) / 8))) and packed.dtype == np.uint8
    np.testing.assert_array_equal(packed, als_module.prepare_als_versions(points, SETTINGS, seed=7))
    masks = np.unpackbits(packed, axis=1, count=len(points)).astype(bool)
    # Different headings -> different versions, each keeps a plausible share
    assert len({m.tobytes() for m in masks}) == 3
    assert all(0.2 < m.mean() < 1.0 for m in masks)


def test_prepare_versions_match_the_emulation(sphere):
    points, _, _ = sphere
    packed = als_module.prepare_als_versions(points, SETTINGS, seed=3)
    rng = np.random.default_rng(3)
    base = rng.uniform(0.0, 180.0)
    for v in range(3):
        lines = als_module.strip_flight_lines(points, heading=base + v * 60.0, phase=rng.uniform(0.0, 1.0))
        expected = emulate_als(points, ALSConfig(flight_lines=lines))
        np.testing.assert_array_equal(np.unpackbits(packed[v], count=len(points)).astype(bool), expected)


def test_mixing_lookup_and_probability():
    rng = np.random.default_rng(0)
    masks = rng.random((2, 1003)) < 0.5
    packed = np.packbits(masks, axis=1)
    inds = rng.choice(1003, 400, replace=False)

    # Off: no random draw
    np.random.seed(1)
    assert als_module.prepared_als_mask(packed, inds, make_config(augment_als=0.0)) is None
    assert np.random.rand() == np.random.RandomState(1).rand()

    # On: every result is one of the versions at these indices, both versions are used, ratio is respected
    np.random.seed(2)
    config = make_config(augment_als=0.4)
    picks = []
    for _ in range(500):
        m = als_module.prepared_als_mask(packed, inds, config)
        if m is None:
            picks.append(-1)
            continue
        matches = [v for v in range(2) if np.array_equal(m, masks[v, inds])]
        assert len(matches) == 1
        picks.append(matches[0])
    picks = np.array(picks)
    assert np.mean(picks >= 0) == pytest.approx(0.4, abs=0.07)
    assert (picks == 0).sum() > 50 and (picks == 1).sum() > 50


def test_mixing_keeps_sample_when_too_few_points_remain():
    packed = np.packbits(np.zeros((1, 64), dtype=bool), axis=1)
    np.random.seed(0)
    assert als_module.prepared_als_mask(packed, np.arange(64), make_config(augment_als=1.0)) is None


def test_preparation_step_caches(tmp_path, sphere):
    points, _, _ = sphere
    calls = []

    def counting_prepare(p, settings, seed=0):
        calls.append(seed)
        return als_module.prepare_als_versions(p, settings, seed=seed)

    prepare = load_method('prepare_als', prepare_als_versions=counting_prepare)
    clouds = {'cloud_a': points, 'cloud_b': points[::2]}

    ds = fake_dataset(tmp_path, clouds)
    prepare(ds)
    assert len(ds.als_versions) == 2 and len(calls) == 2
    assert ds.als_versions[0].shape[0] == 2
    assert len(list((tmp_path / 'input_0.040' / 'als').glob('*.npy'))) == 2

    # Second load: from the cache
    ds2 = fake_dataset(tmp_path, clouds)
    prepare(ds2)
    assert len(calls) == 2
    for a, b in zip(ds.als_versions, ds2.als_versions):
        np.testing.assert_array_equal(a, b)

    # Other settings: recomputed
    ds3 = fake_dataset(tmp_path, clouds, augment_als_versions=3)
    prepare(ds3)
    assert len(calls) == 4 and ds3.als_versions[0].shape[0] == 3


@pytest.mark.parametrize('split, config', [('validation', {}), ('training', {'augment_als': 0.0})])
def test_preparation_step_skipped(tmp_path, sphere, split, config):
    points, _, _ = sphere
    prepare = load_method('prepare_als')
    ds = fake_dataset(tmp_path, {'cloud_a': points}, split=split, **config)
    prepare(ds)
    assert ds.als_versions is None
    assert not (tmp_path / 'input_0.040').exists()


def test_prepared_hook_uses_the_cloud_versions(tmp_path, sphere):
    points, _, _ = sphere
    clouds = {'cloud_a': points[::3], 'cloud_b': points}
    ds = fake_dataset(tmp_path, clouds)
    load_method('prepare_als')(ds)
    hook = load_method('als_augmentation')
    inds = np.arange(0, len(points), 5)
    np.random.seed(0)
    mask = hook(ds, 1, inds)
    versions = np.unpackbits(ds.als_versions[1], axis=1, count=len(points)).astype(bool)
    assert any(np.array_equal(mask, versions[v, inds]) for v in range(2))
