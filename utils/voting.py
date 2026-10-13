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


def knn_voting(search_tree, labels, query_points, knn=5, eps=1e-3, chunk=1_000_000):
    """
    Label of every query point = weighted vote of its knn nearest labelled points.
    Weight of a neighbour = (distance of the nearest neighbour + eps) / (its distance + eps).
    Ties go to the smallest label.

    :param search_tree: sklearn KDTree (or any tree with query(points, k)) over the labelled points
    :param labels: int[M] labels of the points in the tree
    :param query_points: float[N, 3]
    :return: labels[N] (same dtype as labels)
    """
    labels = np.asarray(labels)
    values, label_idx = np.unique(labels, return_inverse=True)
    label_idx = label_idx.ravel()
    n_classes = len(values)
    k = min(knn, len(labels))
    out = np.empty(len(query_points), dtype=labels.dtype)
    for start in range(0, len(query_points), chunk):
        dist, ind = search_tree.query(query_points[start:start + chunk], k=k)
        dist = dist + eps
        weights = dist.min(axis=1, keepdims=True) / dist
        m = ind.shape[0]
        flat = (np.arange(m)[:, None] * n_classes + label_idx[ind]).ravel()
        votes = np.bincount(flat, weights=weights.ravel(), minlength=m * n_classes).reshape(m, n_classes)
        out[start:start + chunk] = values[votes.argmax(axis=1)]
    return out
