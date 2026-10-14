"""
Evaluation metrics (utils/evaluation.py).
"""
from types import SimpleNamespace

import numpy as np
import pandas
import pytest

from utils.evaluation import confusion, ground_truth, scores

LABELS = [0, 1, 2, 3]  # 0 = ignored


def test_confusion_and_scores():
    truth = np.array([1, 1, 1, 2, 2, 3, 0, 0])
    pred = np.array([1, 1, 2, 2, 2, 1, 1, 2])
    C, classes, left_out = confusion(truth, pred, LABELS, ignored_labels=[0])
    assert classes.tolist() == [1, 2, 3] and left_out == 2
    np.testing.assert_array_equal(C[:, :3], [[2, 1, 0], [0, 2, 0], [1, 0, 0]])
    r = scores(C, ["rail", "wire", "mast"])
    # rail: tp 2, true 3, predicted 3 -> 2 / 4; wire: 2 / 3; mast: 0 / 1
    assert r["IoU"] == pytest.approx({"rail": 50.0, "wire": 200 / 3, "mast": 0.0})
    assert r["mIoU"] == pytest.approx((50 + 200 / 3 + 0) / 3)
    assert r["accuracy"] == pytest.approx(100 * 4 / 6) and r["points"] == 6


def test_chunks_give_the_same_matrix():
    rng = np.random.default_rng(0)
    truth, pred = rng.integers(0, 4, 10_000), rng.integers(0, 4, 10_000)
    a = confusion(truth, pred, LABELS, [0])[0]
    b = confusion(truth, pred, LABELS, [0], chunk=777)[0]
    np.testing.assert_array_equal(a, b)


def test_unknown_labels_are_left_out_and_absent_classes_not_averaged():
    truth = np.array([1, 1, 9])          # 9: not a class of the model
    pred = np.array([1, 1, 1])
    C, classes, left_out = confusion(truth, pred, LABELS, [0])
    assert left_out == 1
    r = scores(C, ["rail", "wire", "mast"])
    assert r["mIoU"] == 100.0 and r["IoU"]["wire"] is None


def test_ground_truth_from_column_or_mapping():
    cloud = pandas.DataFrame({"system": [1, 2], "component": [16, 47]})
    cfg = SimpleNamespace(label_mapping=None, pred_selected="system")
    assert ground_truth(cloud, cfg).tolist() == [1, 2]
    from railtwin_common.labels import LabelMapping
    s = LabelMapping.load().level("system").labels
    cfg = SimpleNamespace(label_mapping="obm_mapping.json", label_level="system", label_column="component")
    assert ground_truth(cloud, cfg).tolist() == [s["rail"], s["wire"]]
    assert ground_truth(cloud, cfg, label_column="system").tolist() == [1, 2]
