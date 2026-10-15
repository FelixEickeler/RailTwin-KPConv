# Labels: from fine components to training classes

The clouds keep fine labels (components). A mapping file turns them into the classes a model trains on, when a cloud
is loaded. Code: `railtwin_common/labels/` (`LabelMapping`, `LabelChain`), `datasets/labels.py`.

## Ingredients: the mapping file

`railtwin_common/labels/obm_mapping.json` (from RailTwin-Virtualizer `ecppm2022_simple_mapping.json`):

| part | what it is |
|---|---|
| `synthetic` / `physical` | source names (IFC types / annotation names) → component |
| `component_labels` | 64 components → id. These ids are stored in the clouds (column `component`) |
| `systems` + `system_labels` | component → one of 21 systems: the built-in level `system` |
| `levels` (optional) | own class sets |

Every mapping has the levels `component` (the components themselves) and `system`.

## Recipe: a level

A level says which component goes into which class:

```json
"levels": {
  "kpconv16": {
    "labels":     {"ignored": 0, "ground": 1, "rail": 2, "mast": 3},
    "components": {"catenary-pole": "mast", "cantilever": "mast"},
    "default":    "ignored",
    "ignored":    ["ignored"]
  }
}
```

- `labels`: classes and their ids.
- `components`: component → class. A component not listed goes to the class of the same name, if there is one.
- `default`: class of all remaining components (optional).
- `ignored`: classes that are not trained (optional).

Checks: a class name that is not in `labels` (typo) is an error. A component without a class is an error, unless
there is a `default`.

## Cooking: the config

```python
label_mapping = 'obm_mapping.json'   # a bare name is looked up in railtwin_common/labels, else a path
label_level   = 'kpconv16'
label_column  = 'component'          # column with the fine ids in the clouds
```

Loading a cloud maps `label_column` through a lookup table to the class ids of the level. The classes, their ids and
the ignored classes come from the level and are saved in the run's `parameters.json`; prediction and `evaluate.py`
use them. The cache folder name holds the mapping, level and a hash of the table, so caches of different recipes do
not mix.

Without `label_mapping` the classes come from `dataset_labels` / `dataset_ignored` and the column `system`.

## Chains

If the input data has its own labels, put a step before:

```python
label_mapping = [['input_mapping.json', 'consolidated'],
                 ['obm_mapping.json', 'kpconv16']]
```

The steps are linked by name: the classes of one step are component names of the next.

## In Python

```python
from railtwin_common.labels import LabelMapping, LabelChain

mapping = LabelMapping.load("obm_mapping.json")
level = mapping.level("system")                      # level.labels, level.ignored, level.names
ids = mapping.remap(component_ids, "system")         # numpy array of any shape
cid = mapping.component_id("Ausleger", "synthetic")  # source name -> component id

chain = LabelChain.load([["input_mapping.json", "consolidated"], ["obm_mapping.json", "system"]])
ids = chain.remap(input_ids)
```
