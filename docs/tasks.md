# Open tasks: bigger simplifications

From the simplicity review (2026-10-14). The small fixes of that review are done. These tasks change behaviour, remove
options or touch many files, so each needs a decision first.

## 1. ALS emulation: fewer modes and settings

`utils/als_simulation.py` has 460 lines, the configs have 12 `augment_als*` settings and `emulate_als.py` has 18 flags.
Goal: about 280 lines, 7 settings and 10 flags.

- [ ] **Remove online mode.** It emulates per input sphere, so occluders outside the sphere cast no shadow; the
  prepared mode does not have this problem.
  - Remove `plane_to_flight_line` and `als_augmentation` from `als_simulation.py`.
  - Remove the online branch in `datasets/common.py` (`als_augmentation`).
  - Remove the config keys `augment_als_mode`, `augment_als_planes` and `augment_als_random_heading`.
  - Tests: about 11 tests in `tests/test_als_augmentation.py` go.
- [ ] **Flight lines only as strips.** Remove `random_flight_lines` and `FlightLine.pitch`, plus the CLI flags
  `--offsets`, `--pitch`, `--random-lines` and `--random-heading`.
- [ ] **Remove voxel mode.** Remove `ALSConfig.voxel_size`, `augment_als_voxel` and `--voxel`. Keep
  `voxel_subsample` for `--spacing`.
- [ ] **Decide: multiple returns.** Removing them saves about 30 lines and 4 settings, but they are the one way to
  see through vegetation along the track. Suggestion: keep them.
- [ ] **Defaults in one place.** Today they live in the config classes, in `getattr(config, …, default)` in
  `als_simulation.py`, and in `common.py`. Put them in the base `Config` (as was done for the other options) and
  read them directly.
- [ ] **Remove `ALSConfig.reference_height`** (no caller sets it). Accept only `np.random.Generator`.
- [ ] **Docs:** describe `emulate_als.py` as for inspecting and exporting; training mixes in the prepared versions.

## 2. Configs: shorter, less duplication

`config_railtwin.py` and `config_railtwin_tl.py` share about 60 identical new lines. Only `dataset_task`, `in_radius`,
`learning_rate`, `checkpoint_gap`, `batch_num`/`val_batch_num` and the unfreeze schedules really differ.

- [ ] **ALS block.** Shorten it to `augment_als = 0.0` plus a comment pointing to the defaults (after task 1).
- [ ] **Decide: inheritance?** Option: `config_railtwin_tl.RailTwinConfig(config_railtwin.RailTwinConfig)` overrides
  only the values that differ. It saves about 230 lines, but you no longer see the whole config in one file, unlike
  KPConv's one file per config.
  - `Config.save()` must then walk the class hierarchy, or the inherited keys are missing from `parameters.json`.
  - Suggestion: keep two files.

## 3. Compatibility paths

- [ ] **Decide: `pendulum_legacy` rotation.** It is only needed to reproduce old trainings exactly. If none will be
  re-run, remove it from `datasets/rotations.py`, together with `augment_rotation_yaw` and the tests that check it
  is identical to the old code. Saves about 10 lines and 2 tests.
- [ ] **Decide: the `-ft` / `--fine_tune` flag** in `train_rtib3+_tl.py`. It duplicates `--pretrained <results>`.
  Remove it when no script uses it any more.
- [ ] **Decide: the `RailTwin_static` import fallback** in `train_railtwin.py`. Today it silently switches to
  `rtib3p` when `RailTwin_static` is missing. Import `rtib3p` directly once it is clear that `RailTwin_static` will
  not come back.

## 4. Smaller clean-ups

- [ ] `RTIB3pCustomBatch`/`RTIB3pCollate` (`datasets/rtib3p.py`, about 140 lines) is a copy of the S3DIS batch class,
  made to remove the dependency on S3DIS. Option: move it once to `datasets/common.py`, so S3DIS and RailTwin share
  it. This touches the upstream file `S3DIS.py`, which makes future upstream syncs harder.
- [ ] Name the 8.9 m reference radius of the `expected_N` scaling (7 dataset files), e.g. `REFERENCE_RADIUS`.
- [ ] Docker: should `.env.example` keep `SSH_PASSWORD=slice` (the old PyCharm login), or leave it empty so everyone
  has to set their own?
