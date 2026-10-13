"""
Tests for the Mix3D augmentation (datasets/mixing.py) and its dataset hook (PointCloudDataset.mix3d).
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from datasets.mixing import mix_spheres

KEYS = ('points', 'features', 'labels', 'point_inds', 'input_inds', 'cloud_inds', 'scales', 'rots')


def batch(n_spheres=6, seed=0):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(50, 200, n_spheres)
    return {
        'points': [rng.normal(size=(s, 3)).astype(np.float32) for s in sizes],
        'features': [rng.normal(size=(s, 1)).astype(np.float32) for s in sizes],
        'labels': [rng.integers(0, 5, s) for s in sizes],
        'point_inds': [int(i) for i in range(n_spheres)],
        'input_inds': [rng.integers(0, 10000, s) for s in sizes],
        'cloud_inds': [int(i % 2) for i in range(n_spheres)],
        'scales': [np.ones(3, dtype=np.float32) * i for i in range(n_spheres)],
        'rots': [np.eye(3, dtype=np.float32) * (i + 1) for i in range(n_spheres)],
    }


def test_off_draws_no_random_numbers():
    b = batch()
    np.random.seed(0)
    assert mix_spheres(b, 0.0) is b
    assert np.random.rand() == np.random.RandomState(0).rand()


def test_always_merges_pairs_and_keeps_all_points():
    b = batch(n_spheres=5)
    np.random.seed(0)
    m = mix_spheres(b, 1.0)
    assert len(m['points']) == 3  # (0,1), (2,3), 4
    assert sum(len(p) for p in m['points']) == sum(len(p) for p in b['points'])
    np.testing.assert_array_equal(m['points'][0], np.concatenate([b['points'][0], b['points'][1]]))
    np.testing.assert_array_equal(m['labels'][1], np.concatenate([b['labels'][2], b['labels'][3]]))
    np.testing.assert_array_equal(m['input_inds'][0], np.concatenate([b['input_inds'][0], b['input_inds'][1]]))
    # Per-sphere values keep the first sphere of a pair
    assert m['point_inds'] == [0, 2, 4] and m['cloud_inds'] == [0, 0, 0]
    np.testing.assert_array_equal(m['rots'][1], b['rots'][2])
    # Every merged sample stays consistent
    for p, f, l, i in zip(m['points'], m['features'], m['labels'], m['input_inds']):
        assert len(p) == len(f) == len(l) == len(i)


def test_probability():
    np.random.seed(1)
    merged = 0
    for _ in range(500):
        merged += 2 - len(mix_spheres(batch(n_spheres=2), 0.3)['points'])
    assert merged / 500 == pytest.approx(0.3, abs=0.06)


def test_single_sphere_unchanged():
    b = batch(n_spheres=1)
    assert mix_spheres(b, 1.0) is b


def load_hook():
    source = (Path(__file__).resolve().parents[1] / 'datasets' / 'common.py').read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'PointCloudDataset')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'mix3d')
    namespace = {'np': np, 'mix_spheres': mix_spheres}
    exec(compile(ast.Module([fn], []), 'common.py', 'exec'), namespace)
    return namespace['mix3d']


@pytest.mark.parametrize('split, calibrating, prob, merged', [
    ('training', False, 1.0, True),
    ('training', True, 1.0, False),      # paused while calibrating the batch sizes
    ('validation', False, 1.0, False),
    ('training', False, 0.0, False),
])
def test_hook(split, calibrating, prob, merged):
    hook = load_hook()
    b = batch(n_spheres=4)
    ds = SimpleNamespace(set=split, calibrating=calibrating, config=SimpleNamespace(augment_mix3d=prob))
    out = hook(ds, *[b[k] for k in KEYS])
    assert len(out) == len(KEYS)
    assert (len(out[0]) == 2) == merged
