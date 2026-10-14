# RailTwin-KPConv — Code Review and Improvement List

File-by-file review of the code this fork adds to KPConv-PyTorch (state: branch `update-upstream-kpconv`, 2026-10-13).
Line numbers ignore CR line endings and may drift as files are edited.

Tags: **bug** = wrong result or crash · **risk** = breaks under some conditions · **cleanup** = maintainability.
✔ = verified by reading the code during the review. Unmarked items come from the reviewers' reading and should be
checked before relying on them.

---

## 0. Top priorities

These affect **results** (training or prediction quality), not just convenience.

| # | Where | Problem | Status |
|---|---|---|---|
| 1 | `test_railtwin.py:34` + `datasets/inference_dataset.py:306` | `augment_color = 0.0` → `rand() > 0.0` is always true, so **all input features are zeroed during prediction**. The network sees only geometry at inference, while training kept the features on 80 % of spheres. Fix: `augment_color = 1.0` and skip feature dropout for non-training sets. | ✔ open |
| 2 | `datasets/rtib3p*.py` (~line 162) | Building the **validation** dataset sets the shared `config.batch_num = val_batch_num`. The training sampler is calibrated afterwards, so **training runs at batch 6 instead of 11**, while the saved `parameters.json` still says 11. | ✔ mitigated: `config_railtwin.py` uses `batch_num = val_batch_num = 6` (what actually ran), so the overwrite has no effect. The datasets themselves are unchanged |
| 3 | `datasets/rotations.py` (`'pendulum'`) | Rotation **axis** `u` is nearly horizontal (φ ≤ ±2.5°), but the angle α is uniform in 0..2π. Clouds are rotated by *any* angle about a horizontal axis, including upside down. The intent was a random yaw plus a small tilt. Also `deg/360·π/2` = deg/4, so "10°" is really 2.5°, and `augment_rotation_yaw` is not defined anywhere (bare `except` → always default). | ✔ **fixed**: `'pendulum'` is now a bounded swing (`augment_rotation_max_tilt`, default 10°). The original rotation is kept bit-identical as `'pendulum_legacy'`. Changes training with the RailTwin configs |
| 4 | `datasets/rtib3p*.py` features | `in_features_dim = 2` → network input `[1, above_ground]`; **intensity is loaded and augmented but never used**. Dims 4/5 give a wrong width in the potential path; the random path has no dim-2 branch. | ✔ open: confirm intent |
| 5 | `utils/new_config.py` save/load | `save()` dumps only the subclass's own class attributes. Inherited defaults and instance values (`num_classes`, `labels` set by the dataset, …) are lost. `load()` mutates class attributes and returns a new object that callers ignore. JSON made `lr_decays` keys strings → **LR decay silently off after reload**. | lr_decays ✔ fixed, rest open |
| 6 ✔ fixed | `utils/trainer_railtwin.py` validation | **mIoU is never logged** (print commented out). `worst_Class` names are shifted when a low label is ignored. `argpartition(IoUs, 5)` fails with ≤ 5 classes. `error_count` never incremented. | open |
| 7 ✔ fixed | `utils/predictor.py` / `test_railtwin.py` | Prediction cache is keyed only by file stem and date, so another checkpoint on the same day **reuses stale predictions**, and same-named files in subfolders collide. float32 cast of georeferenced coordinates loses ~0.5 m precision. | open |
| 8 | Fine-tuning (`train_rtib3+_tl.py`, trainer) | Could train for zero epochs, the layer-unfreeze schedule was commented out, the log was overwritten per stage, and the last epoch was never checkpointed. | ✔ **fixed**, see §1 |

---

## 1. Already done on this branch

- **Unfreeze schedule** (`utils/config_railtwin_tl.py`):
  ```python
  unfreeze_schedule = [
      ['head_softmax', 11],
      ['head_mlp', 20],
      ['decoder_blocks', 20],
      ['encoder_blocks', None],
  ]
  ```
  `ModelTrainer.train_with_unfreeze_schedule()` freezes the network and unfreezes one part per stage. Stage lengths
  are counted from the start of fine-tuning; a resumed run skips the stages it has finished. An empty list trains
  everything. Tested with a stub trainer (fresh, resumed mid-stage, stopped early).
- **`train_rtib3+_tl.py`:**
  - `--pretrained <run|chkp>` loads weights only and starts at epoch 0.
  - `-ptp` resumes weights, optimizer and epoch, using that run's `parameters.json`/`.txt`.
  - `-ft` was removed (2026-10-14); use `--pretrained <run>`.
- **Fine-tune loading:**
  - Only weights with matching shapes are loaded, so `head_softmax` re-initialises when the class count changes (e.g. 11 → 16).
  - `map_location` is set.
- **`train()`:**
  - The training log is appended instead of truncated.
  - The PID file is no longer deleted on the last epoch, which used to trigger the kill-signal `break` before the epoch finished.
  - The final checkpoint is taken from the current state (it was a stale `save_dict`, or a `NameError` after 0 epochs).
  - The TensorBoard writer stays open between stages.
  - `train()` now returns whether it finished.
- **Smaller trainer fixes:**
  - The checkpoint `epoch` is the next epoch to train, so resuming no longer repeats an epoch or applies its LR decay twice.
  - `lr_decays.get(1)`.
  - `eval_params[0]` (unpacking crashed with a single batch).
  - `load_subsampled_clouds(work_dir)` in the validation error handler.
- **`new_config.load_json`:** `lr_decays` keys are converted back to `int`.

- **Input features** (`datasets/features.py`):
  - `input_features = ['above_ground']` in the config (names: `above_ground`, `intensity`, `height`; `[]` = geometry only);
  - feature dropout only for training;
  - `drop_features` / `--drop-features` for prediction;
  - old configs keep their features (tested bit-identical).
- **Validation logging:**
  - `Test/mIoU` and per-class `Test/IoU/<class>`;
  - correct class names for the worst class and the plots;
  - no crash with ≤ 5 classes;
  - failing batches are skipped and counted;
  - ASCII TensorBoard tags (renamed: `Test/IoU_worst`, `Test/IoU_worst5`, `Output_Loss/*`, `Reg_Loss/*`).
- **Prediction:**
  - `-m` takes a run folder, a checkpoints folder or a `.tar` file;
  - output mirrors the input subfolders;
  - unreadable files are skipped;
  - the cache is keyed by input file and checkpoint;
  - local coordinates (float32 precision);
  - unlabelled clouds work;
  - `map_location`, no CUDA calls on CPU;
  - vectorised kNN vote (`utils/voting.py`, identical result, ~3× faster).

---

## 2. Datasets

### 2.1 How the 9 rtib3p variants differ (basis for merging them)

| file | labels | ignore | data → cache | split (validation = stem contains) | config | notes |
|---|---|---|---|---|---|---|
| `rtib3p` | 10 classes (instance, wire, catenary-mast, platform, vegetation, noise-barrier, sleeper, rail, tree, rail-bed, stuff) | instance | `<dataset_path>` → `workdir` | `_4_unified` | shared, mutated | has Sampler/Collate/Batch |
| `rtib3p_11classes` | 13 labels (ignored, stuff, ground, …, noise-barrier) = 12 trainable | ignored | `synth/rtib3+_11classes` → `synth/train/11classes` | `_quintile_4` | shared, mutated | globs twice and assumes the same order |
| `rtib3p_16classes` | 17 labels (adds platform-asset, cantilever, contact-wire, dropper) | ignored | `synth/rtib3+_16classes` → `synth/train/16classes` | `_quintile_4` | shared, mutated | globs twice |
| `_add_1` | 16-class | ignored | `pysical/16classes` → `…/16classes_1` | q3, q4; **excludes q1, q2** | shared | |
| `_add_2` | 16-class | ignored | same → `…/16classes_2` | q3, q4; excludes q2 | deepcopy (but `num_classes` written to caller) | |
| `_add_3` | 16-class | ignored | same → `…/16classes_2` | q3, q4; no exclusion | deepcopy | used by `train_rtib3+_tl.py` |
| `rtib3p_real` | 11-class dict | **instance (not in dict → KeyError)** | `<dataset_path>` → `workdir` | `_3_/_4_unified` | forces `batch_num = 12` | ✔ cannot be constructed; ~800 commented lines |
| `rtib3p_reworked` | 10-class | instance | `<dataset_path>` → `workdir` | `_4_unified` | not mutated | no dim-2 branch |
| `test_dataset` | identical to `reworked` except the name | | | | | not an inference dataset |

**Proposed merged class** `RailTwinDataset(config, set, use_potentials, load_data, preset=... | fields)`:
- `labels: dict`, `ignored: list` (validated against `labels`).
- `data_root`, `cache_dir`.
- `glob='**/*.ply'`, a regex split (`train_parts`, `val_parts`, `test_parts`).
- `features=('above_ground', 'intensity')`, `label_field='system'`.
- `expected_N`, `pot_dl_ratio`, `name`.
- Never mutate the caller's config. Keep `batch_num` per instance.
- Presets: `rtib10`, `synth11`, `synth16`, `phys16(train, val)`, `real`.

### 2.2 `rtib3p.py` (and all variants)
- bug ✔ — the validation set writes `config.batch_num` (priority 2). In random mode, `num_centers` vs the `epoch_inds` width also mismatch.
- bug — feature width is wrong for `in_features_dim` 4/5 → build features from a spec and assert the width.
- risk — unsorted glob, substring split, no check that the train/val splits are non-empty.
- risk — the cache is keyed only by stem and `dl`: no source mtime, no `pot_dl`; only the KD-tree's existence is checked. Calibration pkl files sit in the data folder without a split/feature key (train and val, and add_1/2/3, share them).
- risk — feature dropout also runs for validation/test, which adds noise to validation mIoU.
- risk — labels are looked up per point in a Python dict, so an unknown raw label raises a KeyError mid-epoch.
- cleanup — `prepare_S3DIS_ply` is dead. Helpers come from `from utils.mayavi_visu import *`; import them explicitly.

### 2.3 Variant-specific
- `_11classes` / `_16classes`: the second glob is assumed to align with `cloud_names` → glob once. With `**`, the stem as cache key collides across subfolders.
- `_add_*`: substring `_quintile_1` also matches `_quintile_1x`. Files without a quintile tag silently go to training.
- `_add_2` / `_add_3`: `num_classes` is written to the caller's config, not the deep copy → the sampler sees `None` or a stale value.
- `rtib3p_real`: KeyError on construction ✔; forces `batch_num = 12`; `pot_dl` is r/10 vs r/50 with the same cache filename (first run wins).
- `rtib3p_reworked` = `test_dataset`: duplicates → delete `test_dataset.py`.

### 2.4 `inference_dataset.py`
- bug ✔ — features zeroed (priority 1). Skip colour augmentation for inference.
- bug — features hard-wired to columns `[:, :1]`, `[:, 1:2]`; `random_item` builds them differently (colours + height).
- bug — the coarse potentials tree is cached as `{stem}_coarse.pkl` without `in_radius`/`pot_sample_mult`.
- bug — on non-convergence: `plt.show()` then `1/0` → blocks or crashes headless runs; raise `RuntimeError` instead.
- risk — batch/neighbour limits are written into the prediction folder, and `force_redo=True` recalibrates every file.

### 2.5 `datasets/common.py`
- bug ✔ — `'pendulum'` rotation (priority 3). Done: `'pendulum'` = random heading + swing of at most `augment_rotation_max_tilt` degrees; the original behaviour is `'pendulum_legacy'` (bit-identical). Covered by `tests/test_rotations.py`.

---

## 3. Training

### 3.1 `utils/trainer_railtwin.py`
- bug — mIoU is not printed or logged → log `Test/mIoU`.
- bug — `worst_Class` indexes the class array with ignored labels removed but is looked up by label value → wrong class names.
- bug — `np.argpartition(IoUs, 5)` fails with ≤ 5 classes and is swallowed by the bare `except ValueError`.
- bug — `error_count` is never incremented, so the first `IndexError` silently returns partial predictions.
- risk — `net.reg_loss` is an int or a CUDA tensor → `np.nanmean` fails with deformable blocks. Use `float(...)`.
- risk — confusion-plot axis labels follow JSON insertion order, while rows follow sorted `label_values`.
- risk — `config.saving = False` → `Path(None)` crash. The default `saving_path = "results"` makes runs share one folder.
- cleanup — TensorBoard tags contain Greek letters (`ιOutput_Loss`, `Test/ΙoU_w5)`) → use ASCII.
- cleanup — accuracy is printed with `%` but is in 0..1.
- cleanup — a full copy of upstream `ModelTrainer`. Subclass it and override `__init__`, `train` and `cloud_segmentation_validation` only.

### 3.2 Train scripts (`train_rtib3+.py`, `train_rtib3+_tl.py`, `train_railtwin.py`)
- bug — `train_rtib3+.py`: `-ft` overrides `-ptp`. `os.path.join('results', …)` mangles the checkpoint path. It looks for `parameters.txt`, but only `.json` is written, so the run config is never reloaded. (Fixed in all three training scripts, 2026-10-14.)
- bug — `train_railtwin.py`:
  - CLI arguments are passed in the wrong order (`dataset` → `working_dir`, which is unused).
  - `chosen_chkp` is unassigned when there is no `checkpoints/` folder.
  - `Path('').exists()` is always true.
  - `config.load` is called without checking the file exists.
- bug — GPU auto-selection reads `utilization.memory` (bandwidth, not MB). It may select several GPUs, but only `cuda:0` is used. It is also copied 3 times → one helper using `memory.used/total`.
- cleanup — redundant `config.save()`. `os.kill(SIGINT)` exits with a traceback.

### 3.3 Configs
- bug — `utils/new_config.py:90-91`: `pot_sample_mult: 50` and `tukey_update_crop: 0.7` are **annotations**, not defaults → `AttributeError` when missing from the JSON.
- bug — `save()` / `load()` round-trip (priority 5): `load(dir)` only looks for `parameters.txt`, and `saving_path` becomes the `.txt` path.
- cleanup — `config_railtwin.py` and `config_railtwin_tl.py` differ in only 7 values → `RailTwinTLConfig(RailTwinConfig)`.
- cleanup — `label_selector` uses `pred_names` / `pred_selected`, which the training configs never define.

### 3.4 `utils/plot_cm.py`
- bug — FN/FP are swapped and one line has an operator-precedence error → `selectivity`, `npv`, `err_rate` are wrong (not displayed today).
- risk — IoU for empty classes differs from `IoU_from_confusions` → table and TensorBoard disagree.
- cleanup — colours use raw counts while the text is normalised. `applymap` is deprecated. plotly/PIL/pandas imports are hard dependencies of the trainer.

---

## 4. Inference

### 4.1 `test_railtwin.py`
- bug ✔ — `augment_color = 0.0` (priority 1).
- bug — for a directory input, `output_path` is the list of input files; for a file input it is `.parent` of that list → crash.
- bug — `-m <file.tar>` leaves `model_path = None`.
- bug — the default `~/results` is never passed through `expanduser()`.
- bug — `cfg.load()` result is ignored → stale `num_layers` / `deform_layers` if the architecture differs from the default.
- risk — recursive scan, but cache and output keyed by stem → collisions. It also picks up its own `*_predicted.ply` and misses `.LAZ`.
- risk — `check_gpu_memory` crashes without `nvidia-smi` or with `-gid 0,1`.
- cleanup — one bad file aborts the batch. `--metadata` and `--test` are unused.

### 4.2 `utils/predictor.py`
- bug — `torch.load` has no `map_location`; `torch.cuda.synchronize` runs on CPU.
- bug — inference requires label columns → unlabelled clouds raise a KeyError.
- bug — the log line is missing its `f` prefix (`{inference_path.name}`).
- bug — float32 precision on georeferenced coordinates → subtract an offset first.
- risk — points never inside a vote sphere keep all-zero probabilities → `argmax` gives `label_values[0]` (possibly the ignored label).
- risk — calibration advances the potentials before voting, and there is no max-epoch guard. Workers re-fork every epoch (`persistent_workers=True`).
- risk — prediction folders older than the newest 4 are deleted automatically.
- perf — `KnnVoting` runs once per point through `pandas.apply`. Vectorised replacement (same weights and tie-break):
  ```python
  def KnnVoting(in_cloud, out_cloud, colname, search_tree=None, knn=5, eps=1e-3, chunk=2_000_000):
      if not isinstance(search_tree, KDTree):
          search_tree = KDTree(in_cloud[["x", "y", "z"]].to_numpy())
      labels, lab_idx = np.unique(in_cloud[colname].to_numpy(), return_inverse=True)
      lab_idx = lab_idx.ravel()
      C = len(labels)
      xyz = out_cloud[["x", "y", "z"]].to_numpy()
      out = np.empty(len(xyz), dtype=labels.dtype)
      for s in range(0, len(xyz), chunk):
          dist, ind = search_tree.query(xyz[s:s + chunk], k=min(knn, len(lab_idx)))
          dist += eps
          w = dist.min(axis=1, keepdims=True) / dist
          M = ind.shape[0]
          flat = (np.arange(M)[:, None] * C + lab_idx[ind]).ravel()
          votes = np.bincount(flat, weights=w.ravel(), minlength=M * C).reshape(M, C)
          out[s:s + chunk] = labels[votes.argmax(axis=1)]
      return out
  ```
  `KnnVoting` lives in the `railtwin_common` submodule, so the change belongs there.

### 4.3 `test_models.py`
- bug — `set=set` passes the builtin `set` type (the upstream variable was removed).
- bug — `RTIB3p` is only bound inside the `if` → `NameError`. The overwrite module's Sampler/Collate are ignored.
- bug — `utils.config.Config.load` reads `.txt`, but RailTwin runs save `.json`.
- bug — `os.chdir(output_path)` runs before the model path is loaded → a relative path breaks.

---

## 5. Utilities and infrastructure

### 5.1 `utils/ply.py`
- bug — `current_element` is undefined if a property appears before an element.
- bug — `element face` does not reset it → the face list property is added to the vertex dtype → `KeyError`. Files with faces crash.
- bug — offsets are always read before vertices, whatever the header order. The offset dtype ignores endianness.
- open — the offset is read but never applied.

### 5.2 Small tools
- `utils/debug_tools.py`: `is_pinned` is never called (always truthy); `obj.volatile` no longer exists; the "size" counts elements, not bytes.
- `log_mem.py`: GPU index 1 is hard-coded; fragile `nvidia-smi` parsing; a new Timer thread is created every second; unused imports.

### 5.3 Docker and docs
- **Versions disagree:**
  - base image is CUDA 11.0;
  - the torch install is an unpinned cu113 build;
  - `environment.yml` pins `torch==1.11.0+cu113` without the extra index and misses pandas/scipy/plotly/….
  - `requirements.txt` includes unrelated packages (fiftyone, boto3, pymongo).
  - Fix: one pinned set of dependencies.
- **Security:** the Dockerfile hard-codes the user password and gives passwordless sudo.
- **Entrypoint:** `exec gosu … /bin/bash` ignores `CMD`, and the TensorBoard line after it never runs.
- **Compose:**
  - no `tty: true`;
  - non-standard `UUID` variable;
  - `DISPLAY` set without the X11 socket;
  - legacy `runtime: nvidia`;
  - `docker-compose.yml.save` is a leftover.
- `cpp_wrappers/compile_wrappers.sh` lost its executable bit.
- **README:**
  - the compose command lacks `DST` and `cd docker`;
  - ports are not documented;
  - the cpp-wrapper build step is missing;
  - the upstream citation and licence notice were removed (MIT requires the notice);
  - typos ("Thomas Hugues" → Hugues Thomas).
- **INSTALL.md** still describes Ubuntu 18.04 / PyTorch 1.4 / CUDA 10.1.
- **`.gitignore`:** `home` and `rtib_tmp` are unanchored.

---

## 6. Suggested order of work

1. **Quick result fixes** (small diffs):
   - priority 1 (inference features);
   - priority 2 (`batch_num` mutation);
   - mIoU logging and the `worst_Class` mapping.
2. **Decide on retraining:** fix the `'pendulum'` rotation and the feature set (intensity). Both change model behaviour.
3. **Merge the 9 datasets** into one parametrised class (§2.1). This removes most dataset bugs at once.
4. **Config:** a proper save/load round-trip (`new_config.py`); `RailTwinTLConfig` as a subclass.
5. **Inference:**
   - robust CLI;
   - cache keys including checkpoint and parameters;
   - coordinate offset;
   - unlabelled clouds;
   - vectorised `KnnVoting` in `railtwin_common`.
6. **Trainer as a subclass of upstream** to simplify future upstream syncs.
7. **Docker and docs:** pinned environment, no hard-coded password, updated README/INSTALL.
