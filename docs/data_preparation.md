# Data preparation

What clouds should look like before training, prediction (`test_railtwin.py`) and evaluation (`evaluate.py`).
Values refer to the real-data config `utils/config_railtwin_tl.py` (`in_radius = 8.5`, `first_subsampling_dl = 0.04`).

## 1. Tiles: overlap of at least `in_radius`

KPConv predicts every point from the sphere of radius `in_radius` around it. A cloud is never given to the network
as a whole: the predictor cuts spheres out of it and averages the predictions of all spheres a point lies in.

At the border of a tile the spheres are cut off: points within about `in_radius` (8.5 m) of the border see only half
of their surroundings. Their predictions are weaker, and tiles predicted one by one show seams along the borders.

Tiles cut by area without overlap therefore need a **buffer**:

1. Cut the core tiles as now (by area, no overlap).
2. For prediction, cut each tile with a buffer of at least `in_radius` around its core (e.g. 10 m for
   `in_radius = 8.5`): the buffer points come from the neighbouring tiles.
3. Predict the buffered tile and keep the predictions of the core points only. Each point then belongs to exactly
   one core and was predicted with its full surroundings.

Mark the core points when cutting (e.g. a column `core` with 1 for core and 0 for buffer points), so the core can be
selected after the prediction without the tile bounds. For evaluation, only core points should be counted, otherwise
the buffer points are counted twice and the weaker border predictions lower the scores.

Tiles at the end of a track have no neighbour on one side; nothing can be done there.

**Training** does not need the buffer: the training spheres are random, and a few spheres at tile borders do not
matter. Larger tiles have fewer borders, which helps a little.

**Changing `in_radius`** (e.g. a model with a larger radius) needs a buffer of at least the new radius.

## 2. Tile size

The network works on spheres, so the GPU memory does not depend on the tile size. RAM does: the predictor keeps the
whole tile, its subsampled version and a KD-tree in memory. Measured on synthetic rail clouds: about 9 GB fixed
(mostly the DataLoader workers, fewer with smaller `input_threads`) plus about 170 bytes per input point.

| free RAM | points per tile, about |
|---|---|
| 16 GB | 4e7 |
| 32 GB | 1.4e8 |
| 64 GB | 3e8 |

Files with many columns need more. Tiles of up to a few 1e7 points are safe on any machine with 32 GB.

## 3. Point fields

- `x`, `y`, `z`: georeferenced coordinates are fine (float64). Training and prediction work in local coordinates;
  the predicted cloud keeps its original coordinates.
- Input features named in `input_features` (`datasets/features.py`): point fields such as `above_ground` or
  `intensity`, or `height` (the absolute z). Every cloud needs the fields of the model's `input_features`.
- Labels (training and evaluation):
  - without a label mapping: column `system` with the ids of `dataset_labels`;
  - with `label_mapping` (`railtwin_common/labels`): the fine ids in `label_column` (default `component`), mapped to
    the training classes when the cloud is loaded.
  - Points that should not count get the label of `dataset_ignored` (e.g. `ignored`).
- Prediction: unlabelled clouds are fine. Labels that do not fit the model's classes are not used.

## 4. Files and names

- Training reads `**/*.ply` below `dataset_path`. Prediction and evaluation read `.ply`, `.las`, `.laz` and
  `.feather`.
- Cloud names (file names without folder) must be unique: the cache in `dataset_work_dir` is per name.
- The validation set is chosen by name: clouds whose name contains one of `validation_patterns`
  (e.g. `_quintile_4`). `exclude_patterns` leaves clouds out.
- `dataset_work_dir` must not be below `dataset_path` (clouds found there are skipped).
- After a scikit-learn upgrade, delete the `input_<dl>*` folders in the work dir (the KD-trees are pickled).

## 5. Different sources

Clouds of different sensors (MLS, TLS, ALS) can be mixed. Subsampling to `first_subsampling_dl` (4 cm) brings dense
clouds to about the same density; sparse clouds (ALS) stay sparser. The prediction calibration is stored per density
class (`rtib_tmp/calibration/`), so each source gets its own calibration once and later clouds of the same source
reuse it.
