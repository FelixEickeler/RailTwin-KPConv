"""
Tests for the class-balanced sampler (RTIB3pSampler.__iter__ without potentials) and the unfreeze schedule
(ModelTrainer.train_with_unfreeze_schedule). The methods are loaded from source, without the datasets / C++ wrappers.
"""
import ast
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
nn = torch.nn

ROOT = Path(__file__).resolve().parents[1]


def load_method(path, cls_name, name, namespace):
    source = (ROOT / path).read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    fn.name = '_method'
    exec(compile(ast.Module([fn], []), path, 'exec'), namespace)
    return namespace['_method']


# ----------------------------------------------------------------------------------------------------------------------
# Class-balanced sampler
# ----------------------------------------------------------------------------------------------------------------------

def run_sampler(input_labels, label_values, ignored=(0,), batch_num=6, steps=50):
    iterate = load_method('datasets/rtib3p.py', 'RTIB3pSampler', '__iter__',
                          {'np': np, 'torch': torch, 'warnings': warnings})
    num_classes = len([v for v in label_values if v not in ignored])
    epoch_n = steps * batch_num
    dataset = SimpleNamespace(use_potentials=False,
                              epoch_i=torch.zeros(1, dtype=torch.int64),
                              epoch_inds=torch.zeros((2, epoch_n), dtype=torch.int64),
                              config=SimpleNamespace(batch_num=batch_num, num_classes=num_classes),
                              label_values=np.array(label_values), ignored_labels=list(ignored),
                              label_names=[f'class{v}' for v in label_values],
                              input_labels=input_labels)
    sampler = SimpleNamespace(dataset=dataset, N=steps)
    assert len(list(iterate(sampler))) == steps
    return dataset.epoch_inds.numpy()


def centre_classes(inds, input_labels):
    return np.array([input_labels[c][i] for c, i in inds.T])


def test_balanced_centres():
    rng = np.random.default_rng(0)
    # Class 3 is rare (0.5 %), class 0 is ignored
    clouds = [rng.choice([0, 1, 2, 3], 20000, p=[0.2, 0.5, 0.295, 0.005]) for _ in range(2)]
    np.random.seed(0)
    inds = run_sampler(clouds, [0, 1, 2, 3])
    classes = centre_classes(inds, clouds)
    counts = np.array([np.sum(classes == c) for c in [1, 2, 3]])
    assert np.sum(classes == 0) == 0
    assert counts.min() > 0.25 * len(classes)  # about a third each, despite 0.5 % for class 3


def test_absent_class_does_not_hang():
    rng = np.random.default_rng(1)
    clouds = [rng.choice([1, 2], 5000) for _ in range(2)]  # class 3 has no points at all
    np.random.seed(0)
    with pytest.warns(UserWarning, match='class3 has no points'):
        inds = run_sampler(clouds, [0, 1, 2, 3])
    classes = centre_classes(inds, clouds)
    assert set(np.unique(classes)) == {1, 2}
    assert inds.shape[1] == 300


# ----------------------------------------------------------------------------------------------------------------------
# Unfreeze schedule
# ----------------------------------------------------------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_blocks = nn.Linear(2, 2)
        self.decoder_blocks = nn.Linear(2, 2)
        self.head_mlp = nn.Linear(2, 2)
        self.head_softmax = nn.Linear(2, 2)


class FakeTrainer:
    def __init__(self, net, epoch=0, lr=0.01):
        self.epoch = epoch
        self.optimizer = torch.optim.SGD([{'params': net.parameters()}], lr=lr)
        self.log = []

    def train(self, net, a, b, config):
        trainable = sorted(n for n, m in net.named_children() if next(m.parameters()).requires_grad)
        self.log.append((self.epoch, config.max_epoch, trainable, self.optimizer.param_groups[0]['lr']))
        self.epoch = config.max_epoch
        return True


def schedule_method():
    return load_method('utils/trainer_railtwin.py', 'ModelTrainer', 'train_with_unfreeze_schedule', {'np': np})


def test_gradual_schedule():
    net = Net()
    trainer = FakeTrainer(net)
    config = SimpleNamespace(max_epoch=60, learning_rate=0.01, deform_lr_factor=0.1, lr_decays={}, unfreeze_schedule=[['head_softmax', 11], ['head_mlp', 20],
                                                              ['decoder_blocks', 20], ['encoder_blocks', None]])
    assert schedule_method()(trainer, net, None, None, config)
    assert [(a, b) for a, b, _, _ in trainer.log] == [(0, 11), (11, 31), (31, 51), (51, 60)]
    assert trainer.log[0][2] == ['head_softmax']
    assert len(trainer.log[-1][2]) == 4
    assert config.max_epoch == 60


def test_real_schedule_with_parts_and_lr_factor():
    net = Net()
    trainer = FakeTrainer(net, lr=0.008)
    config = SimpleNamespace(max_epoch=100, learning_rate=0.008, deform_lr_factor=0.1, lr_decays={},
                             unfreeze_schedule=[
        ['head_softmax', 2],
        [['head_mlp', 'decoder_blocks', 'encoder_blocks'], None, 0.1]])
    schedule_method()(trainer, net, None, None, config)
    (e0, m0, parts0, lr0), (e1, m1, parts1, lr1) = trainer.log
    assert (e0, m0, parts0, lr0) == (0, 2, ['head_softmax'], 0.008)
    assert (e1, m1, len(parts1)) == (2, 100, 4)
    assert lr1 == pytest.approx(0.0008)


def test_resume_skips_finished_stages():
    net = Net()
    # Resumed optimizer: its learning rate already contains the decays and the factor of the earlier run
    trainer = FakeTrainer(net, epoch=40, lr=1e-5)
    decays = {i: 0.9 for i in range(1, 100)}
    config = SimpleNamespace(max_epoch=100, learning_rate=0.01, deform_lr_factor=0.1, lr_decays=decays,
                             unfreeze_schedule=[['head_softmax', 2],
                                                [['head_mlp', 'decoder_blocks', 'encoder_blocks'], None, 0.1]])
    schedule_method()(trainer, net, None, None, config)
    assert len(trainer.log) == 1
    epoch, _, parts, lr = trainer.log[0]
    # Initial rate x decays of epochs 1..39 x stage factor, not 1e-5 x 0.1
    assert epoch == 40 and len(parts) == 4 and lr == pytest.approx(0.01 * 0.9 ** 39 * 0.1)
