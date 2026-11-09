---
inclusion: fileMatch
fileMatchPattern: "preprocess/**"
---

# Stage 2: Preprocessing

Folder: `preprocess/`. Takes the per-WAV pulse data from stage 1, keeps pulses that match their class, resamples them, adds features, and normalizes them.

## Processing order for one WAV

Implemented in `preprocess/pipeline.py::process_wav_file`:

1. Run stage 1 (`zc.pipeline.process_wav`). Skip the file if it returns nothing.
2. Filter pulses with `filtering.filter_file_pulses`. The class is the parent folder name. Each pulse must fall inside the class ranges in `pulse_class_filters.json` for `fmin`, `fmax`, `fc`, and `duration`.
3. Resample each accepted pulse onto a fixed time grid with `pchip.pchip_resample_pulses` (`config.PCHIP_TIME_INTERVAL`).
4. Add `slope` and `position` with `features.process_pulse_dataframe`.
5. Z-score the columns in `config.NORMALIZATION_COLUMNS` with `normalize.normalize_dataframe`, using `normalization_stats.json`.
6. `main()` writes one CSV per WAV to `config.OUTPUT_DIR` and accumulates rejection counts per class.

## Normalization statistics

`get_normalization_stats.py` computes mean and population std for each column over every WAV in `config.INPUT_DIR`. It runs the same steps 1 to 4 and combines per-batch stats with the parallel Welford formula in `combine_stats`.

- Re-run it when `NORMALIZATION_COLUMNS`, the filter file, or the input data changes.
- `normalize_dataframe` raises an error if the std is zero or not finite. Do not work around that by adding a small epsilon without asking.
- Current stats in `normalization_stats.json`: `freq_hz` mean 33384.97, std 5695.54; `amplitude` mean 555.32, std 618.11. The file also has `slope` and `position`, which are not normalized today.

## Rules for this stage

- Keep the filter, resample, feature, and normalize steps in the same order in both `pipeline.py` and `get_normalization_stats.py`. If they differ, the stats do not match the data they normalize.
- Any new filter or feature must be added to both scripts, or moved into a shared function that both call.
- Class names must match the keys in `pulse_class_filters.json`. An unknown class raises `ValueError`. Keep that behavior.
- Outputs are in z-score units after this stage. Do not feed them to downstream code as raw frequencies.
- `classifier/inference.py` runs steps 3 to 5 itself (PCHIP, features, normalize) and skips step 2 (class filtering). When you change any of those steps, update inference to match.
