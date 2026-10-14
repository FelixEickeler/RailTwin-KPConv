#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Segmentation losses
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      RailTwin addition
#
#      config.segmentation_loss: weights of the loss terms, e.g. {'cross_entropy': 1.0, 'lovasz': 1.0}
#      config.label_smoothing:   label smoothing of the cross entropy (0 = off)
#      config.class_w:           class weights of the cross entropy (as in KPConv)
#
#      Lovasz-softmax: Berman et al., "The Lovasz-Softmax loss: A tractable surrogate for the optimization of the
#      intersection-over-union measure in neural networks", CVPR 2018. It optimises the IoU per class directly and
#      is commonly combined 1:1 with the cross entropy for LiDAR segmentation (e.g. SalsaNext, PTv3).
#

import torch
import torch.nn as nn


def lovasz_grad(gt_sorted):
    """Gradient of the Lovasz extension of the Jaccard loss w.r.t. sorted errors."""
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    if len(gt_sorted) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax(probas, labels, ignore_index=-1):
    """
    Lovasz-softmax loss, averaged over the classes present in the batch.

    :param probas: float[N, C] class probabilities (softmax output)
    :param labels: int[N] class indices in [0, C), ignore_index is skipped
    """
    valid = labels != ignore_index
    probas, labels = probas[valid], labels[valid]
    if probas.numel() == 0:
        return probas.sum() * 0.0
    losses = []
    for c in range(probas.shape[1]):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg[perm])))
    if not losses:
        return probas.sum() * 0.0
    return torch.stack(losses).mean()


class SegmentationLoss(nn.Module):
    """Weighted sum of cross entropy and Lovasz-softmax, configured by the config (see module header)."""

    TERMS = ('cross_entropy', 'lovasz')

    def __init__(self, config):
        super().__init__()
        weights = config.segmentation_loss
        unknown = set(weights) - set(self.TERMS)
        if unknown:
            raise ValueError(f'Unknown loss terms {sorted(unknown)}, available: {list(self.TERMS)}')
        self.weights = {name: float(w) for name, w in weights.items() if float(w) != 0.0}
        if not self.weights:
            raise ValueError('segmentation_loss needs at least one term with a non-zero weight')

        weight = torch.tensor(config.class_w, dtype=torch.float32) if len(config.class_w) > 0 else None
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, ignore_index=-1, label_smoothing=config.label_smoothing)

    def forward(self, outputs, target):
        """
        :param outputs: float[1, C, N] logits (KPFCNN layout)
        :param target: int[1, N] class indices, -1 = ignored
        """
        total = 0.0
        if 'cross_entropy' in self.weights:
            total = total + self.weights['cross_entropy'] * self.cross_entropy(outputs, target)
        if 'lovasz' in self.weights:
            probas = torch.softmax(outputs[0].transpose(0, 1), dim=1)
            total = total + self.weights['lovasz'] * lovasz_softmax(probas, target[0], ignore_index=-1)
        return total
