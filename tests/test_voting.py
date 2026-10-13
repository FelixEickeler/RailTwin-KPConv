"""
Tests for the vectorised kNN label vote (utils/voting.py) against the original railtwin_common KnnVoting.
"""
import numpy as np
import pandas
import pytest
from scipy.spatial import cKDTree

from utils.voting import knn_voting


def original_knn_voting(in_cloud, out_cloud, colname, search_tree, knn=5):
    """railtwin_common.algorithms.voting.KnnVoting + pandas.apply, as used by the predictor before."""
    dist, ind = search_tree.query(out_cloud[["x", "y", "z"]].to_numpy(), k=knn)
    dist += 0.001
    probabilities = 1 / (dist / (np.min(dist, axis=1)[:, np.newaxis]))
    preds = in_cloud[colname].to_numpy()
    bins = np.zeros(np.max(preds) + 1)

    def _vote(out_row):
        bins.fill(0)
        np.add.at(bins, preds[ind[out_row.name]], probabilities[out_row.name])
        return np.argmax(bins)

    return out_cloud.apply(_vote, axis=1, raw=False).to_numpy()


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_same_result_as_the_original(seed):
    rng = np.random.default_rng(seed)
    sub = rng.uniform(0, 10, (3000, 3))
    labels = rng.integers(0, 8, 3000)
    full = rng.uniform(0, 10, (5000, 3))
    tree = cKDTree(sub)
    in_cloud = pandas.DataFrame(np.c_[sub, labels], columns=['x', 'y', 'z', 'pred']).astype({'pred': int})
    out_cloud = pandas.DataFrame(full, columns=['x', 'y', 'z'])
    expected = original_knn_voting(in_cloud, out_cloud, 'pred', tree)
    np.testing.assert_array_equal(knn_voting(tree, labels, full, knn=5), expected)


def test_chunks_and_sparse_labels():
    rng = np.random.default_rng(3)
    sub = rng.uniform(0, 10, (2000, 3))
    labels = rng.choice([3, 17, 250], 2000)  # sparse label values
    full = rng.uniform(0, 10, (3000, 3))
    tree = cKDTree(sub)
    a = knn_voting(tree, labels, full, chunk=10_000)
    b = knn_voting(tree, labels, full, chunk=7)
    np.testing.assert_array_equal(a, b)
    assert set(np.unique(a)) <= {3, 17, 250}


def test_exact_points_keep_their_label():
    rng = np.random.default_rng(4)
    sub = rng.uniform(0, 100, (500, 3))
    labels = rng.integers(0, 5, 500)
    # Isolated points: the nearest neighbour (distance 0) dominates the vote
    np.testing.assert_array_equal(knn_voting(cKDTree(sub), labels, sub, knn=3), labels)
