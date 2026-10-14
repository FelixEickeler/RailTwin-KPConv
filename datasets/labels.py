#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Training classes from a label mapping (railtwin_common/labels)
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      RailTwin addition. With config.label_mapping set, the clouds keep fine ids (column config.label_column) and
#      are mapped to the training classes when a cloud is loaded, before the subsampling. The cache folder name holds
#      the mappings, levels and a hash of the label table, so caches of different (or edited) mappings do not mix.
#          label_mapping = 'obm_mapping.json'                    one mapping, classes of config.label_level
#          label_mapping = [['input_mapping.json', 'consolidated'],  a chain: input ids -> consolidated components
#                           ['obm_mapping.json', 'system']]          -> training classes
#      Without config.label_mapping the dataset keeps its own classes and reads the 'system' column.
#

import hashlib

import numpy as np

from railtwin_common.labels import LabelChain


def mapped_labels(dataset, config, label_from_json, to_ignore):
    """
    Classes of the dataset: (label name -> id, ignored names). From the mapping if config.label_mapping is set,
    otherwise the dataset's own label_from_json / to_ignore, unchanged.
    """
    dataset.label_mapping = None
    if not config.label_mapping:
        return label_from_json, to_ignore
    chain = LabelChain.load(config.label_mapping, config.label_level)
    chain.lookup_table()  # fails now, not at the first cloud, if a label has no class
    dataset.label_mapping = chain
    level = chain.level
    print(f"Classes from {chain.name}: {len(level.labels)} classes, ignored {level.ignored}")
    return dict(level.labels), list(level.ignored)


def read_labels(dataset, data):
    """Training labels of a cloud as read from file (structured array or DataFrame)."""
    if dataset.label_mapping is None:
        return data['system']
    return dataset.label_mapping.remap(data[dataset.config.label_column])


def cache_folder_name(dataset, dl):
    """'input_0.040', with a label mapping e.g. 'input_0.040_obm_mapping_system_1a2b3c4d' (hash of the label table)."""
    name = 'input_{:.3f}'.format(dl)
    if dataset.label_mapping is not None:
        table = dataset.label_mapping.lookup_table()
        name += f"_{dataset.label_mapping.name}_{hashlib.sha1(table.tobytes()).hexdigest()[:8]}"
    return name
