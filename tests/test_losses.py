"""
Tests for the segmentation losses (models/losses.py). Needs torch.
"""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from models.losses import SegmentationLoss, lovasz_softmax  # noqa: E402


def cfg(**kwargs):
    defaults = dict(class_w=[])
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def batch(n=500, c=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, c, n, generator=g)
    target = torch.randint(0, c, (1, n), generator=g)
    target[0, :20] = -1  # ignored points
    return logits, target


def test_default_is_the_old_cross_entropy():
    logits, target = batch()
    old = torch.nn.CrossEntropyLoss(ignore_index=-1)(logits, target)
    new = SegmentationLoss(cfg())(logits, target)
    assert torch.allclose(old, new)


def test_class_weights_as_before():
    logits, target = batch()
    w = [1.0, 2.0, 0.5, 1.0, 3.0]
    old = torch.nn.CrossEntropyLoss(weight=torch.tensor(w), ignore_index=-1)(logits, target)
    assert torch.allclose(old, SegmentationLoss(cfg(class_w=w))(logits, target))


def test_lovasz_of_hard_predictions_is_one_minus_iou():
    """For one-hot probabilities the Lovasz extension equals the Jaccard loss: mean over present classes of 1 - IoU."""
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 4, 300)
    preds = labels.copy()
    flip = rng.random(300) < 0.3
    preds[flip] = rng.integers(0, 4, flip.sum())
    probas = torch.nn.functional.one_hot(torch.tensor(preds), 4).float()
    ious = []
    for c in range(4):
        inter = np.sum((preds == c) & (labels == c))
        union = np.sum((preds == c) | (labels == c))
        if np.any(labels == c):
            ious.append(inter / union)
    expected = 1 - np.mean(ious)
    assert lovasz_softmax(probas, torch.tensor(labels)).item() == pytest.approx(expected, abs=1e-6)


def test_lovasz_perfect_prediction_is_zero_and_ignores_labels():
    labels = torch.tensor([0, 1, 2, 2, -1, -1])
    probas = torch.nn.functional.one_hot(labels.clamp(min=0), 3).float()
    probas[4] = torch.tensor([0.0, 0.0, 1.0])  # wrong, but ignored
    assert lovasz_softmax(probas, labels).item() == pytest.approx(0.0)


def test_combined_loss_and_gradient():
    logits, target = batch()
    logits.requires_grad_(True)
    loss_fn = SegmentationLoss(cfg(segmentation_loss={'cross_entropy': 1.0, 'lovasz': 0.5}))
    loss = loss_fn(logits, target)
    expected = loss_fn.terms['cross_entropy'] + 0.5 * loss_fn.terms['lovasz']
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad[0, :, :20].abs().sum() == 0  # ignored points get no gradient


def test_label_smoothing_is_used():
    logits, target = batch()
    plain = SegmentationLoss(cfg())(logits, target)
    smooth = SegmentationLoss(cfg(label_smoothing=0.2))(logits, target)
    expected = torch.nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.2)(logits, target)
    assert torch.allclose(smooth, expected) and not torch.allclose(plain, smooth)


def test_lovasz_only_and_all_ignored():
    logits, target = batch()
    loss_fn = SegmentationLoss(cfg(segmentation_loss={'lovasz': 1.0}))
    assert set(loss_fn.terms) == set() and loss_fn(logits, target) > 0
    assert loss_fn(logits, torch.full_like(target, -1)).item() == 0.0


@pytest.mark.parametrize('bad', [{'focal': 1.0}, {'cross_entropy': 0.0}])
def test_bad_configs(bad):
    with pytest.raises(ValueError):
        SegmentationLoss(cfg(segmentation_loss=bad))
