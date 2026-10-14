"""
Training classes from a label mapping (datasets/labels.py).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from datasets.labels import cache_folder_name, mapped_labels, read_labels
from railtwin_common.labels import LabelMapping

OWN = {"ignored": 0, "rail": 1}


def config(**kw):
    return SimpleNamespace(**{"label_mapping": None, "label_level": "system", "label_column": "component", **kw})


def test_without_mapping_nothing_changes():
    dataset = SimpleNamespace(config=config())
    assert mapped_labels(dataset, dataset.config, OWN, ["ignored"]) == (OWN, ["ignored"])
    data = {"system": np.array([1, 0]), "component": np.array([16, 3])}
    np.testing.assert_array_equal(read_labels(dataset, data), [1, 0])
    assert cache_folder_name(dataset, 0.04) == "input_0.040"


def test_classes_and_labels_from_the_mapping():
    obm = LabelMapping.load()
    dataset = SimpleNamespace(config=config(label_mapping="obm_mapping.json"))
    labels, ignored = mapped_labels(dataset, dataset.config, OWN, ["ignored"])
    assert labels == obm.level("system").labels and ignored == ["instance"]
    c = obm.component_labels
    data = {"system": np.array([9, 9]), "component": np.array([c["contact-wire"], c["sleeper"]], dtype=np.uint8)}
    np.testing.assert_array_equal(read_labels(dataset, data), [labels["wire"], labels["sleeper"]])
    assert cache_folder_name(dataset, 0.04).startswith("input_0.040_obm_mapping_system_")


def test_chain(tmp_path):
    import json
    obm = LabelMapping.load()
    path = tmp_path / "input_mapping.json"
    path.write_text(json.dumps({"component_labels": {"none": 0, "wire": 5},
                                "levels": {"consolidated": {"labels": {"unclassified": 0, "contact-wire": 47},
                                                            "components": {"none": "unclassified",
                                                                           "wire": "contact-wire"}}}}))
    dataset = SimpleNamespace(config=config(label_mapping=[[str(path), "consolidated"], ["obm_mapping.json", "system"]],
                                            label_column="input"))
    labels, ignored = mapped_labels(dataset, dataset.config, OWN, ["ignored"])
    assert labels == obm.level("system").labels
    np.testing.assert_array_equal(read_labels(dataset, {"input": np.array([5, 0])}), [labels["wire"], labels["instance"]])
    name = cache_folder_name(dataset, 0.04)
    assert name.startswith("input_0.040_input_mapping_consolidated+obm_mapping_system_")

    # An edited mapping gets its own cache
    data = json.loads(path.read_text())
    data["levels"]["consolidated"]["components"]["wire"] = "unclassified"
    path.write_text(json.dumps(data))
    mapped_labels(dataset, dataset.config, OWN, ["ignored"])
    edited = cache_folder_name(dataset, 0.04)
    assert edited != name

    # Another label column gets its own cache too
    dataset.config.label_column = "other"
    assert cache_folder_name(dataset, 0.04) not in (name, edited)


def test_unknown_level_fails_at_start():
    dataset = SimpleNamespace(config=config(label_mapping="obm_mapping.json", label_level="nope"))
    with pytest.raises(KeyError):
        mapped_labels(dataset, dataset.config, OWN, ["ignored"])
