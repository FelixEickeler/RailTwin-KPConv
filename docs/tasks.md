# Open tasks: simplifications

From the simplicity review (2026-10-14).

## Done

- **ALS emulation:** prepared mode with flight strips only. Online mode, random flight lines, pitch and voxel mode
  are removed. Multiple returns stay, because they let the beam see through vegetation.
  - `utils/als_simulation.py` went from 460 to about 350 lines.
  - The config has one entry, `augment_als`; the other settings have defaults in `utils/new_config.py`.
  - `emulate_als.py` went from 18 to 13 options.
- **Configs:** the defaults of all new options are in the base config. `config_railtwin.py` and
  `config_railtwin_tl.py` stay two complete files (KPConv style: one file shows the whole configuration).
- **`-ft` / `--fine_tune` removed.**
  - Fine-tuning: `train_rtib3+_tl.py --pretrained <run>`.
  - Resuming a run: `-ptp <run>` in every training script. This now works with relative paths as well, and
    resuming loads the run's `parameters.json`.
- **SSH:** `SSH_PASSWORD=slice` stays in `.env.example` for the PyCharm login.

## Open, needs a decision

- [ ] **`pendulum_legacy` rotation.** It is only needed to reproduce old trainings exactly. If none will be re-run,
  remove it from `datasets/rotations.py`, together with `augment_rotation_yaw` and the tests that check it is
  identical to the old code. Saves about 10 lines and 2 tests.
- [ ] **`RailTwin_static` import fallback** in `train_railtwin.py`. Today it silently switches to `rtib3p` when
  `RailTwin_static` is missing. Import `rtib3p` directly once it is clear that `RailTwin_static` will not come back.
- [ ] **`RTIB3pCustomBatch`/`RTIB3pCollate`** (`datasets/rtib3p.py`, about 140 lines) is a copy of the S3DIS batch
  class, made to remove the dependency on S3DIS. Option: move it once to `datasets/common.py`, so S3DIS and RailTwin
  share it. This touches the upstream file `S3DIS.py`, which makes future upstream syncs harder.

## Not done, on purpose

- **A name for the 8.9 m reference radius of `expected_N`:** it would touch the 7 `rtib3p*` dataset files (to be
  reworked when needed), and the comment next to each formula already explains it.
