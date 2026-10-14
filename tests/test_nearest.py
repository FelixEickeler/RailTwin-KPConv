"""
nearest_indices (datasets/common.py) gives the same reprojection indices as the sklearn KDTree query.
"""
import numpy as np
import pytest

KDTree = pytest.importorskip("sklearn.neighbors").KDTree

from datasets.common import nearest_indices  # noqa: E402


def test_same_as_sklearn_query():
    rng = np.random.default_rng(0)
    sub = rng.uniform(0, 10, (5000, 3)).astype(np.float32)
    full = rng.uniform(0, 10, (20000, 3)).astype(np.float32)
    tree = KDTree(sub, leaf_size=10)
    expected = np.squeeze(tree.query(full, return_distance=False)).astype(np.int32)
    result = nearest_indices(tree, full)
    assert result.dtype == np.int32
    np.testing.assert_array_equal(result, expected)
