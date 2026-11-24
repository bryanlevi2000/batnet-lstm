---
inclusion: fileMatch
fileMatchPattern: "preprocess/**"
---

# Stage 2: Preprocessing

Folder: `preprocess/`. Takes the per-file pulse data from stage 1 (WAV or ZC files), keeps pulses that match their class, resamples them, adds features, and normalizes them.

## Processing order for one input file

Implemented in `preprocess/pipeline.py::process_input_file`. It never raises. It returns a dict with `file_path`, `status`, `dataframe`, `rejection_stats` and `error`. `status` is `"processed"`, `"no_pulses"` or `"failed"`.

1. Run stage 1 (`zc.pipeline.process_file`). An exception makes the file `"failed"`, with the exception type and message in `error`. Empty pulses make it `"no_pulses"`.
2. Filter pulses with `filtering.filter_file_pulses`. The class is the parent folder name. Each pulse must fall inside the class ranges in `pulse_class_filters.json` for `fmin`, `fmax`, `fc`, and `duration`.
3. Resample each accepted pulse onto a fixed time grid with `pchip.pchip_resample_pulses` (`config.PCHIP_TIME_INTERVAL`).
4. Add `slope` and `position` with `features.process_pulse_dataframe`.
5. Z-score the columns in `config.NORMALIZATION_COLUMNS` with `normalize.normalize_dataframe`, using `normalization_stats.json`.
6. `main()` writes one CSV per input file to `config.OUTPUT_DIR`, named with the full input file name plus `.csv` (`recording.wav.csv`, `H9140021.26#.csv`), and accumulates rejection counts per class. It writes `config.PREPROCESS_FAILED_FILES_PATH` (`file`, `error`) and `config.PREPROCESS_NO_PULSE_FILES_PATH` (`file`). A file where class filtering removes every pulse is in neither list; it is counted in the rejection summary. A file whose pulses all drop out during resampling counts as `"no_pulses"`.

## Normalization statistics

`get_normalization_stats.py` computes mean and population std for each column over every WAV and ZC file in `config.INPUT_DIR`. It runs the same steps 1 to 4 and combines per-batch stats with the parallel Welford formula in `combine_stats`. It writes `config.STATS_FAILED_FILES_PATH` and `config.STATS_NO_PULSE_FILES_PATH`. Files with no pulses left after class filtering are counted as "empty" and are not listed.

ZC files have NaN amplitude. The stats ignore non-finite values, so ZC files add nothing to the `amplitude` stats. If every file is a ZC file, the script raises an error for `amplitude`. Remove `amplitude` from `NORMALIZATION_COLUMNS` in that case.

- Re-run it when `NORMALIZATION_COLUMNS`, the filter file, or the input data changes.
- `normalize_dataframe` raises an error if the std is zero or not finite. Do not work around that by adding a small epsilon without asking.
- Current stats in `normalization_stats.json`: `freq_hz` mean 33384.97, std 5695.54; `amplitude` mean 555.32, std 618.11. The file also has `slope` and `position`, which are not normalized today.

## Rules for this stage

- Keep the filter, resample, feature, and normalize steps in the same order in both `pipeline.py` and `get_normalization_stats.py`. If they differ, the stats do not match the data they normalize.
- Any new filter or feature must be added to both scripts, or moved into a shared function that both call.
- Class names must match the keys in `pulse_class_filters.json`. An unknown class raises `ValueError`. Keep that behavior.
- Outputs are in z-score units after this stage. Do not feed them to downstream code as raw frequencies.
- `classifier/inference.py` runs steps 3 to 5 itself (PCHIP, features, normalize) and skips step 2 (class filtering). When you change any of those steps, update inference to match.
