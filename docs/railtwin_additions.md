# RailTwin-KPConv — Additions to KPConv-PyTorch

This document lists what RailTwin-KPConv adds to or changes in the upstream
[KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch) repository, and records how the fork was
synchronised with upstream.

## 1. Upstream baseline and synchronisation

| | Commit | Date |
|---|---|---|
| Original fork point | `3a774ff` "adding NPM3D support" | 2022-04-13 |
| Synchronised to | `d19c575` "Correcting -1 neighbor bug for non deformable KPConv" | 2025-10-17 |

The fork was imported as a single commit (`e71b1d8`) and shares no git history with upstream. The fork point
was found by diffing every upstream commit against the shared core files; `3a774ff` is the closest match.
The 29 upstream commits after it were applied as a 3-way patch.

### 1.1 Upstream changes pulled in

| Area | Change |
|---|---|
| `models/blocks.py` | **Bug fix:** `-1` (shadow) neighbour indices now point to the appended fake point (`s_pts.shape[0] - 1`) instead of relying on negative indexing. Affects non-deformable KPConv. |
| `datasets/NPM3D.py`, `datasets/S3DIS.py` | **Bug fix:** `center_point` is copied (`np.copy`) before noise is added. Before this, `+=` wrote the noise into the KD-tree's point buffer and moved the cloud a little on every sample. The noise is also clipped to ±`in_radius / 2`. |
| `datasets/NPM3D.py` | Input features no longer include colours, only height (`in_features_dim` must match). |
| `datasets/S3DIS.py` | More robust repair of corrupt (non-ASCII) lines in the raw S3DIS files. |
| `datasets/SemanticKitti.py` | Epoch index wrap-around; `int64` class indices; clear error when a class has no frames; enough random centres in non-balanced mode. |
| New datasets | `datasets/SensatUrban.py` + `train_SensatUrban.py`, `datasets/Toronto3D.py` + `train_Toronto3D.py` (class weights, validation split fix). |
| `test_models.py`, `plot_convergence.py` | Support for SensatUrban and Toronto3D. |
| `LICENSE.txt` | Upstream MIT licence. |
| `doc/visualization_guide.md` | Typo fix. |

### 1.2 Merge decisions

- **`datasets/S3DIS.py` and `train_S3DIS.py` are back** as plain upstream files, so S3DIS support works
  again (`test_models.py`, `plot_convergence.py` and `visualize_deformations.py` import them).
  `rtib3p.py` is a separate, heavily modified copy of the original `S3DIS.py`. It is not replaced by it.
- **Rail code no longer depends on S3DIS.** The fork had removed the batch and collate code from
  `rtib3p.py`, but the train scripts still imported `S3DISCollate` from the deleted `datasets.S3DIS`.
  `RTIB3pCustomBatch` and `RTIB3pCollate` now live in `datasets/rtib3p.py`. `train_rtib3+.py`,
  `train_rtib3+_tl.py`, `train_railtwin.py`, `test_models.py` (rail branch) and `utils/predictor.py`
  import `RTIB3pSampler` / `RTIB3pCollate` from there.
- **The `center_point` fix was also applied to every fork dataset derived from S3DIS:** `rtib3p.py`,
  `rtib3p_11classes.py`, `rtib3p_16classes.py`, `rtib3p_16classes_add_{1,2,3}.py`, `rtib3p_real.py`,
  `rtib3p_reworked.py`, `test_dataset.py` and `inference_dataset.py` (2 places per file).
  ⚠️ This changes training behaviour. Models trained before this change used clouds that drifted from
  the noise.
- **`test_models.py` conflicts:**
  - The fork's argparse-driven entry point was kept.
  - Upstream's hard-coded `chosen_log = 'results/Log_2024-05-14_21-04-36'` was dropped.
  - The `last_*` list now holds `last_S3DIS`, `last_SensatUrban` and `last_rtib+`.

## 2. RailTwin additions

The changes fall into three groups: data and config loading, custom training, and writing results.
Files named `rtib*` belong to a specific dataset; files named `RailTwin*` / `railtwin*` are generic.

### 2.1 Datasets (`datasets/`)

All of these derive from upstream's `S3DISDataset` (potential-based sphere sampling, `cloud_segmentation`).
The main differences from S3DIS:

- Labels are given as a `name → id` dict (`label_from_json`) and saved into the config
  (`config.labels`, `config.ignored_labels`, `config.dataset_name`). `"ignored"` is the ignored class.
- Every `**/*.ply` file under `self.path` is a cloud. The train/validation split comes from the file name
  (e.g. `_quintile_3`, `_quintile_4` → validation).
- Pre-processed data (subsampled clouds, KD-trees, projections) is cached in `self.work_dir`.

| File | Dataset / purpose |
|---|---|
| `rtib3p.py` | Base RTIB3+ dataset; also provides `RTIB3pSampler`, `RTIB3pCustomBatch` and `RTIB3pCollate`, used by all train scripts, `test_models.py` and `utils/predictor.py`. |
| `rtib3p_11classes.py` | Synthetic RTIB3+, 11 classes (`dataset/synth/rtib3+_11classes`). |
| `rtib3p_16classes.py` | Synthetic RTIB3+, 16 classes (`dataset/synth/rtib3+_16classes`). |
| `rtib3p_16classes_add_1.py` | Physical 16-class data; quintiles 1+2 excluded, 3+4 used for validation. |
| `rtib3p_16classes_add_2.py` | Like `add_1`, but only quintile 2 excluded; deep-copies the config. |
| `rtib3p_16classes_add_3.py` | Like `add_1`, but all quintiles used; deep-copies the config. Used by `train_rtib3+_tl.py`. |
| `rtib3p_real.py` | Real (measured) point clouds. |
| `rtib3p_reworked.py`, `test_dataset.py` | Experimental variants. |
| `inference_dataset.py` | `InferenceDataset` / `InferenceSampler`: a dataset without labels over a single cloud and KD-tree, used by `utils/predictor.py`. |

Paths are hard-coded to the Docker layout (`/home/point_warrior/dataset/...`, see §2.6).
To switch the dataset or split, change the `import ... as RailTwinDatasetECPPM2022` line in the train script.

### 2.2 Changes to shared upstream code

| File | Change |
|---|---|
| `datasets/features.py` | Input features chosen by name: `input_features = ['above_ground']` (names: `above_ground`, `intensity`, `height`; `[]` = geometry only), `in_features_dim = 1 + len(input_features)`. Feature dropout (`augment_color`) only for training. `drop_features` zeroes them, e.g. for prediction on data without them (`test_railtwin.py --drop-features`). |
| `datasets/common.py`, `datasets/rotations.py` | Rotations moved to `random_rotation()`. `augment_rotation = 'pendulum'`: random heading plus a swing of at most `augment_rotation_max_tilt` degrees (default 10) around a random horizontal axis, like a pendulum below its pivot. `'pendulum_legacy'` is the original RailTwin rotation, kept bit-identical for old trainings: a random angle in 0..360° around an almost horizontal axis, which often turns the cloud upside down. |
| `utils/ply.py` | `parse_header` reads an optional `element offset` (3× `f8`) block before the vertices and only collects properties of the `vertex` element. It now returns `(num_points, num_offset, properties)`. The offset is read but not applied yet. |
| `utils/config.py` | Commented-out stub for saving dataset and label parameters. |
| `utils/trainer.py` | Whitespace only. |

### 2.3 Configuration (`utils/`)

| File | Purpose |
|---|---|
| `new_config.py` | Reworked `Config` base class (JSON-aware save and load). |
| `config_railtwin.py` | `RailTwinConfig`: KPFCNN, 15 kernel points, `in_radius` 8.9 m, `first_subsampling_dl` 0.04 m, `in_features_dim` 2, LR 1.2e-2, 350 epochs, `augment_rotation='pendulum'`, batch 11/6. |
| `config_railtwin_tl.py` | Transfer-learning variant: `in_radius` 8.5, LR 0.8e-2, checkpoint gap 25, batch 3/3, `dataset_task='cloud_segmentation'`. |

### 2.4 Training and testing

| File | Purpose |
|---|---|
| `train_railtwin.py` | Generic RailTwin training (uses `datasets.RailTwin_static`, see §3). |
| `train_rtib3+.py` | Trains from scratch on RTIB3+ (16 classes). Arguments: working path, previous training, `fine_tune`, GPU id. |
| `train_rtib3+_tl.py` | Transfer learning on `rtib3p_16classes_add_3`. `--pretrained <run\|chkp>` loads weights only (layers whose shape changed, e.g. `head_softmax`, stay initialised) and starts at epoch 0. `-ptp` resumes a run. Training follows `unfreeze_schedule` in `utils/config_railtwin_tl.py`: head_softmax → head_mlp → decoder → encoder (gradual unfreezing). |
| `utils/trainer_railtwin.py` | `ModelTrainer` with `train_with_unfreeze_schedule()`, TensorBoard logging (`SummaryWriter`, confusion-matrix images through `utils/plot_cm.py`). Can fine-tune from a checkpoint or resume a run (`finetune`, `continue_epoch`). |
| `test_models.py` | Argparse CLI (`log_dir`, `model_path`, `out_dir`, `val`, `dataset_overwrite`). Supports the datasets `rtib+`, `RailTwin` and `-rtib`. |
| `test_railtwin.py` | Inference CLI: `-m` model, `-i` input file or folder, `-o` output, `-gid` GPU. Reads clouds through `railtwin_common` IO. |
| `utils/predictor.py` | `Predictor`: subsamples, runs KPFCNN over `InferenceDataset`, and maps predictions back to the full cloud with `KnnVoting`. Writes PLY files with `railtwin_common`. |

### 2.5 Utilities

| File | Purpose |
|---|---|
| `utils/plot_cm.py` | Confusion matrix and per-class overview as Plotly/Matplotlib figures and image arrays (for TensorBoard). |
| `utils/debug_tools.py` | Lists live CUDA tensors and their sizes. |
| `log_mem.py` | Background thread that logs GPU memory (`nvidia-smi`) into a pandas frame. |

### 2.6 Infrastructure

- **Submodule `railtwin_common`** ([RailTwin-Common](https://github.com/FelixEickeler/RailTwin-Common)):
  IO readers and writers, logger, PCA, KNN voting, track divider and splicer, quadindex.
  Set it up with `git submodule init && git submodule update`.
- **`docker/`**: CUDA 11.0 / Ubuntu 20.04 image, with PyTorch (cu113), pyarrow, laspy and plyfile.
  - The container runs as user `point_warrior`, with UID and GID mapped from the host.
  - Volumes: `SRC` → `~/dataset`, `DST` → `~/workdir`, repo → `~/railtwin_kpconv`.
  - Ports: SSH on 2244, TensorBoard on 6667.
  - Start it with:
    ```bash
    UUID="$(id -u)" GID="$(id -g)" SRC=<data> DST=<workdir> docker-compose up kpconv -d
    ```

## 3. Known issues (already present before the sync)

- `train_railtwin.py` imports `datasets.RailTwin_static` (`RailTwinStatic`, probably the ECPPM 2022 dataset
  with a hard-coded cloud list), which is not in the repository. The script now falls back to
  `datasets.rtib3p.RTIB3p` and prints a warning. (`utils/predictor.py` used to import `S3DISCollate` from the missing `datasets.RailTwin_priming`;
  it now uses `RTIB3pCollate`.)
- `rtib3p_16classes_add_2.py` and `rtib3p_16classes_add_3.py` share the same `work_dir`
  (`.../train/16classes_2`). Their cached pre-processing can overwrite each other.
- Dataset paths are hard-coded absolute paths inside the Docker container.
- `utils/ply.py` reads the PLY `offset` element but does not apply it to the points.
- `docker/docker-compose.yml.save` is a leftover editor backup.

See [`code_review.md`](code_review.md) for a file-by-file review and a list of improvements.

### 2.7 ALS emulation (`utils/als_simulation.py`, `emulate_als.py`)

Emulates what an airborne laser scanner would record of a dense MLS / TLS cloud.

- **Flight lines:** straight lines (`FlightLine`: heading, lateral offset, height, max scan angle, pitch), 100 m
  above the lowest point by default. `random_flight_lines()` draws lines that all see the cloud centre.
- **Visibility:** a slope-tolerant z-buffer in sensor coordinates (along track × scan angle). A point is hidden if
  another point within the beam footprint around its ray is clearly closer. Surfaces seen at a steep angle
  (facades in an oblique view) do not hide themselves.
- **Returns:** up to `max_returns` per beam. Later returns pass through gaps (vegetation), not through closed
  surfaces.
- **Voxel mode** (`voxel_size`): visibility is computed per voxel, which is robust to uneven MLS density.
  `voxel_subsample()` reduces the result to an ALS point spacing.
- **Speed:** about 6 s and 1.5 GB for a 10 M point cloud with 6 flight strips. Long lines are processed in
  chunks with an identical result.

**Training with ALS data: switch it on in the config** (`utils/config_railtwin*.py`, off by default)

```python
augment_als = 0.5               # share of training spheres taken from an ALS version (0 = off)
augment_als_mode = 'prepared'
augment_als_versions = 2        # ALS versions per training cloud, headings spread over 180 deg
augment_als_height = 100.0      # flying height above the lowest point [m]
augment_als_max_scan_angle = 30.0
augment_als_side_overlap = 0.3
```

1. **Preparation step:** when a training dataset is loaded, `PointCloudDataset.prepare_als()` computes the ALS
   versions of every training cloud. Each version is flown in parallel strips over the whole subsampled cloud, so
   every object occludes, including objects outside a later input sphere.
   - The versions are stored as bit masks (about 2.5 MB per 10 M points and 2 versions).
   - They are cached in `work_dir/input_<dl>/als/`. The cache key includes the settings, so changing them
     recomputes the versions.
   - Expect about 6 s per 10 M points and strip set.
2. **Mixing:** in `potential_item` / `random_item`, with probability `augment_als`, a training sphere uses one random
   ALS version (a lookup in the mask). Otherwise it stays the original MLS sphere.
   - It runs before the rotation augmentation.
   - Validation and test are never changed.
   - With `augment_als = 0` nothing is prepared and no random number is drawn, so existing trainings are unchanged.
- **Synthetic data** (generated from design models) is complete (roofs, tree crowns), so the emulation is
  physically right there. Real MLS lacks top surfaces; the emulation can only remove points, not add the roof and
  crown tops a real ALS would record. A sensible start is ALS mixing in the synthetic pre-training
  (`config_railtwin.py`) only.

**Alternative: `augment_als_mode = 'online'`.** Each sphere is emulated on the fly from one random plane of
`augment_als_planes`. It costs CPU time in the data loader, and only the points inside the sphere occlude.
- **Plane keys:**
  - `height` [m] above the lowest point of the sphere;
  - `off_nadir` [deg], the angle under which the sphere centre is seen (alternatively `offset` [m]);
  - optional `heading`, `max_scan_angle` and `pitch` [deg].

**Exporting ALS files** (`emulate_als.py`, for inspection or as a separate dataset):

```bash
python emulate_als.py -i dataset/synth/rtib3+_16classes -o out/ --strips --versions 2 --suffix _als
```

- The options mirror the config: `--strips`, `--side-overlap`, `--random-heading`, `--voxel`, `--spacing` (ALS
  point spacing) and `--returns`, plus single lines via `--offsets` / `--random-lines`.
- All fields are kept and `als_return` is added.
- Writing into the input folder requires `--suffix`.

### 2.8 Tests (`tests/`)

Run them with `python -m pytest tests/` (needs numpy, matplotlib and pytest; torch and the C++ wrappers are
not required).

- `test_rotations.py`:
  - `'pendulum_legacy'` is bit-identical to the original `'pendulum'`;
  - documents its upside-down rotations;
  - checks the bounds of the corrected `'pendulum'`.
- `test_als_augmentation.py`:
  - prepared versions: deterministic, identical to the emulation;
  - the cache is reused and recomputed when the settings change;
  - the preparation is skipped for validation, when ALS is off and in online mode;
  - the mixing ratio and the lookup of the masks;
  - the plane definitions survive a JSON round trip;
  - the probability is honoured, and no random draw happens when the augmentation is off;
  - each plane gets picked;
  - an oblique plane reveals the facing facade;
  - the same seed gives the same result;
  - the dataset hook only runs for training and uses the sphere centre.
- `test_als_simulation.py`: synthetic ground, building and canopy scenes covering:
  - occlusion, shadows and the swath;
  - oblique facades and multiple flight lines;
  - second returns through vegetation, but not through roofs;
  - voxel mode and subsampling;
  - georeferenced coordinates.
