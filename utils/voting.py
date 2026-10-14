#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Transfer labels from the subsampled cloud to the full cloud by a weighted kNN vote
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      RailTwin addition: vectorised version of railtwin_common.algorithms.voting.KnnVoting (same weights and
#      tie-breaking), which ran point by point through pandas.apply.
#

import numpy as np
from scipy.spatial import cKDTree


def knn_voting(points, labels, query_points, knn=5, eps=1e-3, chunk=1_000_000):
    """
    Label of every query point = weighted vote of its knn nearest labelled points.
    Weight of a neighbour = (distance of the nearest neighbour + eps) / (its distance + eps).
    Ties go to the smallest label.

    The tree is a scipy cKDTree queried on all cores: for 1e9 query points about 6 min instead of about 1.5 h with
    the single threaded sklearn KDTree.

    :param points: float[M, 3] labelled points
    :param labels: int[M] labels of the points
    :param query_points: float[N, 3]
    :return: labels[N] (same dtype as labels)
    """
    labels = np.asarray(labels)
    values, label_idx = np.unique(labels, return_inverse=True)
    label_idx = label_idx.ravel()
    n_classes = len(values)
    k = min(knn, len(labels))
    search_tree = cKDTree(points, leafsize=10, balanced_tree=False, compact_nodes=False)
    out = np.empty(len(query_points), dtype=labels.dtype)
    for start in range(0, len(query_points), chunk):
        dist, ind = search_tree.query(query_points[start:start + chunk], k=k, workers=-1)
        dist = dist + eps
        weights = dist.min(axis=1, keepdims=True) / dist
        m = ind.shape[0]
        flat = (np.arange(m)[:, None] * n_classes + label_idx[ind]).ravel()
        votes = np.bincount(flat, weights=weights.ravel(), minlength=m * n_classes).reshape(m, n_classes)
        out[start:start + chunk] = values[votes.argmax(axis=1)]
    return out
