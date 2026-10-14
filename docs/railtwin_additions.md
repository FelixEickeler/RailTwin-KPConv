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
  The cache is in `input_<first_subsampling_dl>/` (e.g. `input_0.020/`) and is rebuilt when files are missing.
  The KD-trees are sklearn pickles: after a scikit-learn upgrade, delete that folder once.
  The reprojection of validation clouds onto the subsampled points uses `nearest_indices` (`datasets/common.py`),
  same result as the sklearn query, on all cores.

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
| `datasets/common.py`, `datasets/rotations.py` | Rotations moved to `random_rotation()`. `augment_rotation = 'pendulum'`: random heading plus a swing of at most `augment_rotation_max_tilt` degrees (default 10) around a random horizontal axis, like a pendulum below its pivot. The original RailTwin rotation (a random angle in 0..360° around an almost horizontal axis, often upside down) was removed on 2026-10-14; it is in the git history as `'pendulum_legacy'`. Unknown modes raise an error. |
| `utils/ply.py` | `parse_header` reads an optional `element offset` (3× `f8`) block before the vertices and only collects properties of the `vertex` element. It now returns `(num_points, num_offset, properties)`. The offset is read but not applied yet. |
| `utils/config.py` | Commented-out stub for saving dataset and label parameters. |
| `utils/trainer.py` | Whitespace only. |

### 2.3 Configuration (`utils/`)

| File | Purpose |
|---|---|
| `new_config.py` | Reworked `Config` base class (JSON-aware save and load). |
| `config_railtwin.py` | `RailTwinConfig`: KPFCNN, 15 kernel points, `in_radius` 8.9 m, `first_subsampling_dl` 0.04 m, `in_features_dim` 2, LR 1.2e-2, 350 epochs, `augment_rotation='pendulum'`, batch 6/6. |
| `config_railtwin_tl.py` | Transfer-learning variant: `in_radius` 8.5, LR 0.8e-2, checkpoint gap 25, batch 3/3, `dataset_task='cloud_segmentation'`. |

### 2.4 Training and testing

| File | Purpose |
|---|---|
| `train_railtwin.py` | Generic RailTwin training (uses `datasets.RailTwin_static`, see §3). |
| `train_rtib3+.py` | Trains from scratch on RTIB3+ (16 classes). Arguments: `-o` results folder, `-ptp` run to resume, `-gid` GPU id, `-r` sphere radius. |
| `train_rtib3+_tl.py` | Transfer learning on `rtib3p_16classes_add_3`. `--pretrained <run\|chkp>` loads weights only (layers whose shape changed, e.g. `head_softmax`, stay initialised) and starts at epoch 0. `-ptp` resumes a run. Training follows `unfreeze_schedule` in `utils/config_railtwin_tl.py`: head_softmax → head_mlp → decoder → encoder (gradual unfreezing). |
| `utils/trainer_railtwin.py` | `ModelTrainer` with `train_with_unfreeze_schedule()`, TensorBoard logging (`SummaryWriter`, confusion-matrix images through `utils/plot_cm.py`). Can fine-tune from a checkpoint or resume a run (`finetune`, `continue_epoch`). Checkpoints are found by `utils/checkpoints.py` (newest `*.tar` of a run folder, usually `current_chkp.tar`), the same rule for resuming, fine-tuning and prediction. |
| `test_models.py` | Argparse CLI (`log_dir`, `model_path`, `out_dir`, `val`, `dataset_overwrite`). Supports the datasets `rtib+`, `RailTwin` and `-rtib`. |
| `test_railtwin.py` | Inference CLI: `-m` model, `-i` input file or folder, `-o` output, `-gid` GPU. Reads clouds through `railtwin_common` IO. |
| `utils/predictor.py` | `Predictor`: subsamples, runs KPFCNN over `InferenceDataset`, and maps predictions back to the full cloud with a weighted 5-NN vote (`utils/voting.py`, scipy `cKDTree` on all cores: about 6 min instead of 1.5 h for 1e9 points). The KD-tree is passed to `InferenceDataset` directly, not pickled. Writes PLY files with `railtwin_common`. |

### 2.4b Training options (from the literature review, `docs/literature_review_training.md`)

| Option | Where | What |
|---|---|---|
| `segmentation_loss = {'cross_entropy': 1.0, 'lovasz': 1.0}`, `label_smoothing` | `models/losses.py`, both RailTwin configs | Cross entropy (with `class_w`, label smoothing) plus Lovász-softmax, which optimises the IoU per class. `{'cross_entropy': 1.0}` reproduces the old loss exactly. |
| `balance_classes = True` | both RailTwin configs, train scripts | Training spheres are centred per class instead of by potentials (the KPConv author's recommendation for rare classes). Validation keeps potentials. Classes without points are skipped instead of hanging the sampler. |
| `-r / --in-radius` | `train_rtib3+.py`, `train_rtib3+_tl.py` | Overrides the sphere radius for experiments (e.g. 2.5 / 4 / 6 m vs 8.9 m). It is saved to `parameters.json`, so prediction uses the same value. |
| `augment_mix3d = 0.3` | `datasets/mixing.py`, both RailTwin configs | Mix3D: two training spheres of a batch are overlaid into one sample (out-of-context augmentation, +0.7–1.5 mIoU measured with KPConv). Paused during the batch calibration, training only, 0 = off. |
| `--schedule real` | `train_rtib3+_tl.py`, `unfreeze_schedule_real` | Fine-tuning on real data: 2 epochs head warm-up, then the whole network at 0.1× learning rate. `--schedule gradual` (default) keeps the layer-by-layer unfreezing. Stages may list several parts and an optional learning rate factor (relative to `learning_rate` with the decays so far, also after a resume). |

### 2.5 Utilities

| File | Purpose |
|---|---|
| `utils/plot_cm.py` | Confusion matrix and per-class overview as Plotly/Matplotlib figures and image arrays (for TensorBoard). |
| `utils/debug_tools.py` | Lists live CUDA tensors and their sizes. |
| `log_mem.py` | Background thread that logs GPU memory (`nvidia-smi`) into a pandas frame. |

### 2.6 Infrastructure

- **Submodule `railtwin_common`** ([RailTwin-Common](https://github.com/FelixEickeler/RailTwin-Common)):
  IO readers and writers, logger, PCA, KNN voting, track divider and splicer, quadindex.
  Set it up with `git submodule init && git submodule update`. KPConv uses its PLY / LAS reading, the PLY writer
  and the logger for prediction (`test_railtwin.py`, `utils/predictor.py`); training does not need it.
  - **Updated 2026-10-14** (RailTwin-Common `main`): the writers keep float64 coordinates (before, predictions of
    georeferenced clouds were written as float32, up to ~0.1 m off) and no longer change the caller's DataFrame.
    Unknown PLY columns are kept when reading. The logger is Python `logging` (no spdlog). It works with numpy 2
    and pandas 2, has a `pyproject.toml` and 24 passing tests.
- **`docker/`**: Ubuntu 22.04, Python 3.10, PyTorch 2.3.1 with CUDA 12.1 (RTX 30xx / 40xx). One pinned
  `requirements.txt`: numpy 1.26 (the C++ wrappers build with `numpy.distutils`, which numpy 2 removed),
  scikit-learn 1.0.2 so cached KD-trees still load.
  - **Settings:** in `docker/.env` (from `.env.example`): data and work folder, uid / gid, ssh password and / or public key.
  - **Start-up:** the C++ wrappers are compiled on the first start, and TensorBoard runs on host port 6667.
  - **Access:** ssh (e.g. PyCharm remote interpreter) with the password and / or key from `docker/.env`; no password is built into the image.
  - **Start it with:**
    ```bash
    cd docker && cp .env.example .env && docker compose up -d --build
    ```
  - **Tested** on an RTX 4090 with a short training, fine-tuning (`--pretrained`, `--schedule real`) and
    prediction run on synthetic data.

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

- **Flight lines:** parallel strips that cover the whole cloud (`strip_flight_lines`), 100 m above the lowest point
  by default. Each strip is a straight `FlightLine` (heading, lateral offset, height, max scan angle).
- **Visibility:** a slope-tolerant z-buffer in sensor coordinates (along track × scan angle). A point is hidden if
  another point within the beam footprint around its ray is clearly closer. Surfaces seen at a steep angle
  (facades in an oblique view) do not hide themselves.
- **Returns:** up to `max_returns` per beam. Later returns pass through gaps (vegetation), not through closed
  surfaces.
- **Point spacing:** `voxel_subsample()` reduces the result to an ALS point spacing (`emulate_als.py --spacing`).
- **Speed:** about 6 s and 1.5 GB for a 10 M point cloud with 6 flight strips. Long lines are processed in
  chunks with an identical result.

**Training with ALS data: switch it on in the config** (`utils/config_railtwin*.py`, off by default)

```python
augment_als = 0.5               # share of training spheres taken from an ALS version (0 = off)
```

The emulation settings (`augment_als_versions`, `_height`, `_max_scan_angle`, `_side_overlap`, `_footprint`,
`_returns`, `_min_points`) have defaults in `utils/new_config.py` and can be overridden in the config.

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

**Exporting ALS files** (`emulate_als.py`, for inspection or as a separate dataset):

```bash
python emulate_als.py -i dataset/synth/rtib3+_16classes -o out/ --versions 2 --suffix _als
```

- The options mirror the config: `--versions`, `--height`, `--heading`, `--max-scan-angle`, `--side-overlap`,
  `--footprint`, `--returns`, plus `--spacing` (ALS point spacing).
- All fields are kept and `als_return` is added.
- Writing into the input folder requires `--suffix`.

### 2.8 Tests (`tests/`)

Run them with `python -m pytest tests/` (needs numpy, matplotlib and pytest; torch and the C++ wrappers are
not required).

- `test_rotations.py`:
  - all modes give proper rotations, `'vertical'` keeps the z-axis;
  - `'pendulum'` is bounded by `augment_rotation_max_tilt`, with heading and swing direction covering the full
    circle;
  - unknown modes raise an error.
- `test_als_augmentation.py`:
  - prepared versions: deterministic, identical to the emulation;
  - the cache is reused and recomputed when the settings change;
  - the preparation is skipped for validation and when ALS is off;
  - the mixing ratio and the lookup of the masks, no random draw when the augmentation is off;
  - the dataset hook only runs for training and uses the versions of the right cloud.
- `test_als_simulation.py`: synthetic ground, building and canopy scenes covering:
  - occlusion, shadows and the swath;
  - oblique facades and multiple flight lines;
  - second returns through vegetation, but not through roofs;
  - voxel subsampling and chunked processing;
  - georeferenced coordinates.
