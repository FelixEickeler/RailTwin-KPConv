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
- **`pendulum_legacy` removed:** training is on real data now. The old rotation is in the git history; unknown
  `augment_rotation` values raise an error instead of silently not rotating.
- **Datasets merged** (2026-10-14): the nine `rtib3p*.py` / `test_dataset.py` files (about 10 000 lines) are one
  `datasets/railtwin.py`. Path, work dir, split and classes are config options; checked to give the same data.
- **`train_railtwin.py`** (2026-10-15) imports `datasets.railtwin.RailTwinDataset` directly; the fallback for the
  missing `datasets.RailTwin_static` is gone.
- **`--votes`** (2026-10-15) for `test_railtwin.py` and `evaluate.py`; default 100 as before.
- **Prediction calibration cached** (2026-10-15) per model settings and density class instead of per cloud.

## Not done, on purpose

- **`RailTwinCustomBatch`/`RailTwinCollate`** (`datasets/railtwin.py`) stays a copy of the S3DIS batch class. Sharing it
  would mean changing the upstream file `S3DIS.py` and make future upstream syncs harder.
- **A name for the 8.9 m reference radius of `expected_N`:** the comment next to the formula
  (`datasets/railtwin.py`, `datasets/inference_dataset.py`) explains it.
