"""
Run overview for TensorBoard (utils/run_overview.py): HParams entry and text overview.
"""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensorboard")
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from utils.new_config import Config  # noqa: E402
from utils.run_overview import class_table, describe_start, log_run_overview, run_hparams  # noqa: E402


def fake_dataset():
    return SimpleNamespace(label_values=np.array([0, 1, 2]), label_to_names={0: "unclassified", 1: "rail", 2: "mast"},
                           ignored_labels=[0], input_labels=[np.array([0, 1, 1, 2]), np.array([1, 2, 2, 2])],
                           files=["a.ply", "b.ply"], path="/data/real")


def config():
    c = Config()
    c.dataset = "rtib3p"
    c.architecture = ["simple", "resnetb", "resnetb_deformable_strided", "nearest_upsample", "unary"]
    c.input_features = ["intensity", "above_ground"]
    return c


def test_class_table_counts_training_points():
    assert class_table(fake_dataset()) == [(0, "unclassified", True, 1), (1, "rail", False, 3), (2, "mast", False, 4)]


def test_hparams_are_plain_values():
    net = torch.nn.Linear(3, 2)
    h = run_hparams(config(), net, fake_dataset(), "finetune: synth_run")
    assert all(isinstance(v, (bool, int, float, str)) for v in h.values())
    assert h["classes/trained"] == "rail, mast" and h["classes/num_trained"] == 2
    assert h["inputs/features"] == "intensity, above_ground"
    assert h["model/deformable"] is True and h["model/parameters"] == 8
    assert h["train/start"] == "finetune: synth_run"


def test_written_to_the_run_folder(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    writer = SummaryWriter(tmp_path)
    log_run_overview(writer, config(), torch.nn.Linear(3, 2), fake_dataset())
    writer.close()
    events = EventAccumulator(str(tmp_path)).Reload()
    assert set(events.PluginTagToContent("hparams")) == {"_hparams_/experiment", "_hparams_/session_start_info",
                                                          "_hparams_/session_end_info"}
    assert "Run/Overview/text_summary" in events.Tags()["tensors"]


def test_describe_start():
    assert describe_start(None, False) == "scratch"
    assert describe_start("/w/run_a/checkpoints/current_chkp.tar", True) == "finetune: run_a"
    assert describe_start("/w/run_a/checkpoints/current_chkp.tar", False) == "resume: run_a"
