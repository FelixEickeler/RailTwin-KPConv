# Agent map — RailTwin-KPConv

Terse map for coding agents. Repo = KPConv-PyTorch fork (Hugues Thomas) + RailTwin additions for railway point
cloud semantic segmentation. Long human docs: `docs/railtwin_additions.md`. Open/closed tasks: `docs/tasks.md`. Data prep (tile buffer ≥ in_radius, fields, names): `docs/data_preparation.md`. Label mappings: `docs/labels.md`.

## Rules (from owner)

- Work on `main` only. No branches. Push only when asked. Submodule `railtwin_common` push first, then repo.
- Train scripts stay separate files (`train_rtib3+.py`, `train_rtib3+_tl.py`, `train_railtwin.py`). Shared logic → `datasets/` / `utils/`.
- Real data focus. KPConv original backend only (no KPConvX). Rotation mode name = "pendulum".
- Config style: each RailTwin config file shows whole config (`utils/config_railtwin*.py` not subclassed).
- Upstream files (`datasets/S3DIS.py`, `NPM3D.py`, `Toronto3D.py`, `SemanticKitti.py`, `SensatUrban.py`, `ModelNet40.py`, `train_<upstream>.py`, `models/`, `kernels/`, `cpp_wrappers/`) = touch minimal, keep upstream sync easy.
- Line endings: user git `core.autocrlf=true`; LF edits fine.

## Run / test

- Docker image `railtwin-kpconv:cu121` (CUDA 12.1, PyTorch 2.3.1, numpy 1.26 — C++ wrappers need `numpy.distutils`, so no numpy 2). Build/compose: `docker/`. Settings `docker/.env` (from `.env.example`): data dir → `~/dataset`, work dir → `~/workdir`, repo → `~/railtwin_kpconv`, SSH 2244 (PyCharm interpreter), TensorBoard host 6667.
- C++ wrappers compile on first container start.
- Tests local (no torch-cpp/sklearn; some skip): `~/miniconda3/envs/calybra_ml/bin/python -m pytest tests/ -q -p no:cacheprovider`
- Tests full: `docker run --rm -v $PWD:/home/point_warrior/railtwin_kpconv railtwin-kpconv:cu121 bash -c "cd /home/point_warrior/railtwin_kpconv && python3 -m pytest tests -q -p no:cacheprovider"`
- Library tests: same, `cd railtwin_common && python3 -m pytest -q -p no:cacheprovider`.
- Train/predict scripts end with `os.kill(SIGINT)` → exit 130 normal.

## Entry points (top level)

| file | does |
|---|---|
| `train_rtib3+.py` | train from scratch, synthetic data, `utils/config_railtwin.py`. `-o` work dir, `-ptp` resume run, `-r` in_radius, `-gid` |
| `train_rtib3+_tl.py` | fine-tune real data, `utils/config_railtwin_tl.py`. `--pretrained <run>` weights only (shape-mismatch layers re-init), `-ptp` resume, `--schedule gradual\|real` (unfreeze) |
| `train_railtwin.py` | generic, `RailTwinDataset` + `utils/config_railtwin.py`, picks free GPUs via nvidia-smi |
| `test_railtwin.py` | predict: `-m` run/ckpt, `-i` cloud/folder, `-o` out, `--drop-features`, `--votes` (default 100), `--recalibrate`. Writes `<cloud>_predicted.ply` (col `preds_<pred_selected>`). `resolve_model`, `setup_cfg` reused by evaluate |
| `evaluate.py` | full-res eval vs labels: `-m`, `-i`, `-n` test set, `--label-column`, `--votes`, `--recalibrate`, `--save-predictions`. Out `<run>/evaluation/<n>/evaluation.{json,csv}` + TB run `<run>/logdir/eval_<n>` |
| `test_models.py` | old KPConv tester; RailTwin runs load `parameters.json`, upstream `parameters.txt` |
| `emulate_als.py` | offline ALS emulation of .ply (flight strips), writes new clouds |
| `train_<S3DIS\|NPM3D\|...>.py`, `plot_convergence.py`, `visualize_deformations.py` | upstream |

## Data flow

`.ply` (x,y,z,above_ground,intensity,system [+component]) → `RailTwinDataset` loads, maps labels (opt.), grid-subsamples (`first_subsampling_dl`), caches → sphere sampling by potentials (or class-balanced) → augment (rotation, scale, noise, Mix3D, ALS) → `RailTwinCollate` builds KPConv batch (neighbors/pools via C++) → `KPFCNN` → `SegmentationLoss` → `ModelTrainer`.
Predict: `Predictor` subsample → `InferenceDataset` votes (`--votes`, default 100) → subsampled preds → `knn_voting` to full cloud.

## datasets/

| file | key |
|---|---|
| `railtwin.py` | `RailTwinDataset(config, set, use_potentials)`, `RailTwinSampler` (+`calibration`, class-balanced `__iter__`), `RailTwinCustomBatch`, `RailTwinCollate`. Config-driven: `dataset_path` (recursive `**/*.ply`, sorted), `dataset_work_dir` (cache + calibration pkl), `validation_patterns`, `exclude_patterns`, `dataset_labels`/`dataset_ignored` or `label_mapping`. Works on deepcopy of config; writes `labels`, `ignored_labels`, `num_classes`, `dataset_name` into caller config too. Duplicate cloud names = error; clouds under work dir skipped |
| `common.py` | upstream `PointCloudDataset` + RailTwin: `nearest_indices` (scipy cKDTree all cores, validation reprojection), `prepare_als`/`als_augmentation` (cache `work_dir/input_<dl>/als`), Mix3D hook, `grid_subsampling`, `batch_neighbors` |
| `labels.py` | `mapped_labels` (classes from `LabelChain`), `read_labels` (raw cloud → class ids; no mapping → `system` col), `cache_folder_name` (`input_<dl>[_<chain>_<hash(table+label_column)>]`) |
| `inference_dataset.py` | `InferenceDataset(config, cloud_path, search_tree)` single cloud, potentials, `InferenceSampler` |
| `features.py` | `feature_names`, `select_features`, `stack_features` — `input_features` names (`above_ground`, `intensity`, `height`), `drop_features` |
| `rotations.py` | `random_rotation(mode, max_tilt)` modes `none`, `vertical`, `all`, `pendulum`; unknown → ValueError |
| `mixing.py` | `mix_spheres` Mix3D (training only, off during calibration) |
| others | upstream datasets |

Cache folder `dataset_work_dir/input_<dl>/`: `<cloud>.ply` (subsampled; mapped ids stored in col `system`), `<cloud>_KDTree.pkl` (sklearn pickle → sklearn pinned 1.0.2; delete folder after sklearn upgrade), `_coarse_KDTree.pkl`, `_proj.pkl`, `als/`.

## utils/

| file | key |
|---|---|
| `config.py` / `new_config.py` | base `Config` (upstream txt / RailTwin json). All RailTwin option defaults live here. `save()` → `parameters.json` (class + instance attrs), `load()` |
| `config_railtwin.py` / `config_railtwin_tl.py` | `RailTwinConfig` full configs (synth scratch / real fine-tune): paths, 16 classes, arch, `in_radius 8.9`, `first_subsampling_dl`, batch 6/6, `segmentation_loss {'cross_entropy':1,'lovasz':1}`, `balance_classes`, `unfreeze_schedule(_real)`, `augment_rotation 'pendulum'`, `augment_mix3d 0.3`, `augment_als`, `label_mapping/label_level/label_column` |
| `trainer_railtwin.py` | `ModelTrainer(net, config, chkp_path, finetune, tb_logdir)`: `train`, `train_with_unfreeze_schedule` (stages `[parts, epochs, lr_factor]`, LR = base × decays × factor), `validation` (TB scalars `Test/*`, IoU % ), `log_overview`. Checkpoint dict: epoch, model, optimizer, saving_path, best_mIoU |
| `run_overview.py` | TB HParams session summaries (no experiment summary → columns = all runs' scalars) + `Run/Overview` text: classes w/ training point share, settings, arch |
| `predictor.py` | `Predictor(cfg)`: `run_on_cloud(df, cache_name)` local float32 coords (offset floor(min)), subsample, infer, cache `rtib_tmp/<date>_predictions/<name>_<model_key>_predictions.ply`, `knn_voting` back. Cloud labels not matching model classes → placeholder |
| `voting.py` | `knn_voting(points, labels, query, knn=5)` weighted vote, scipy cKDTree `workers=-1`, chunked |
| `evaluation.py` | `confusion` (chunked, ignored/unknown left out), `scores` (IoU %, mIoU over present classes, acc), `ground_truth`, `report_markdown`, `training_hparams`, `write_tensorboard` |
| `checkpoints.py` | `find_checkpoint(path)` newest `*.tar` (run dir, `checkpoints/`, or file) |
| `als_simulation.py` | ALS emulation: flight strips (`strip_flight_lines`), multi-return (`flight_line_returns`), `emulate_als`, prepared versions as bit masks (`prepare_als_versions`, `prepared_als_mask`, `prepared_als_settings`) |
| `metrics.py`, `ply.py`, `tester.py`, `trainer.py`, `visualizer.py`, `mayavi_visu.py` | upstream |
| `plot_cm.py` | confusion matrix / overview figures for TB |

## models/

`architectures.py` (`KPFCNN`: `encoder_blocks`, `decoder_blocks`, `head_mlp`, `head_softmax` — unfreeze part names), `blocks.py` upstream, `losses.py` (`SegmentationLoss`: CE (+label smoothing, class_w) + Lovász-softmax, `lovasz_softmax`).

## railtwin_common/ (git submodule, own repo FelixEickeler/RailTwin-Common, shared by other projects)

- Package root = repo root (`pyproject.toml` `package-dir {"railtwin_common" = "."}`); import `railtwin_common.common...`. Extras `algorithms`, `plot`, `test`.
- IO: `common.io.read(path|str)`, `write(path, df, options)`, `scandir_supported`. Readers ply/las/laz (laspy, formats 0–5, world coords → local + offset)/feather; writers ply/feather never mutate caller df; both apply IOOptions coords (local/shift/dtype; `write()` default = local float32 + offset). `IOOptions.do_nothing()` keeps float64 coords. Offset = `attrs["metadata"]["offset"]` structured `[(x,y,z)]` shape (1,) via `header.as_offset`. `attrs["category_mapping"]`.
- `labels`: `LabelMapping.load("obm_mapping.json")` levels `component`/`system`/custom (`"levels"` in JSON), `remap(ids, level)`, `component_id(name, "synthetic"|"physical")`; `LabelChain.load([[file, level], ...])` steps linked by class name. `labels/obm_mapping.json` 64 components (from RailTwin-Virtualizer ecppm2022_simple).
- `algorithms`: `QuadIndex` (2D grid index), PCA, minimal distance; `common.logger.RailTwinLogger` (python logging "RT-Logger").

## Gotchas

- Labels in caches are mapped ids; cache key includes mapping hash + label column. ALS cache not per mapping.
- `config.labels`/`ignored_labels`/`num_classes` are set by the dataset at runtime and saved in `parameters.json`; prediction/eval rely on them.
- Resume (`-ptp`) loads run `parameters.json`; old runs w/o `dataset_*` keys fall back to script config defaults.
- Validation dataset sets `batch_num = val_batch_num` on its own config copy only.
- `Predictor(cfg, num_votes=100, recalibrate=False)`: votes are part of the prediction cache key. ~20 votes ≈ 100 (HT); fewer = faster.
- Prediction calibration (batch/neighbor limits) cached in `rtib_tmp/calibration/*.pkl`, key + `_density<class>` (median pts per in_radius sphere, steps 2^(1/4); ±1 class reused). Was redone per cloud (45 s – 6 min).
- Prediction RAM (measured, synthetic MLS): ~9 GB fixed (workers) + ~170 B/input point → ~5e8 points max on 93 GB. GPU already works on spheres; no tiling yet.
- `rtib_tmp/` (prediction cache) gitignored.
