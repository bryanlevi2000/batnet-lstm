---
inclusion: fileMatch
fileMatchPattern: "zc/**"
---

# Stage 1: Zero-Crossing (ZC) Extraction

Folder: `zc/`. Turns a WAV recording or an Anabat ZC file into time-ordered zero-crossing points, then groups them into FM pulses.

## Input file types

`zc/input_files.py` decides what is an input file, for every stage. `iter_input_files(input_dir)` walks a folder recursively in sorted order and yields:

- WAV files: names ending in `.wav`.
- ZC files: names ending in `.zc` or `#`, for example `H9140021.zc` and `H9140021.26#`.

Anything else is ignored. Matching is case-insensitive. Do not add per-stage file discovery; call `iter_input_files`.

`write_file_lists(failed, no_pulses, failed_path, no_pulses_path)` writes the failed-files CSV (`file`, `error`) and the no-pulse-files CSV (`file`). Both are written even when empty.

## Processing order for one file

Implemented in `zc/pipeline.py`. `process_file(path)` is `process_points(load_points(path))`.

`load_points` picks the loader by file name and returns a DataFrame with columns `time_us`, `freq_hz`, `amplitude`:

- WAV: `load_wav_points`.
  1. Load with `librosa.load(sr=None, mono=True)`.
  2. Denoise with `denoise.denoise_audio` using `config.NOISE_FACTOR` and `config.NOISE_STRIDE`.
  3. Convert to int16 (`* 32767`). ZC extraction expects this scale.
  4. Extract zero crossings with `zc_extraction.extract_zc` (`config.DIVRAT`, `config.HYSTERESIS_PERCENTILE`).
- ZC file: `zc_loader.load_zc_points`. See the next section. Denoising and `zc_extraction` are not used.

`process_points` is the same for both file types:

1. Filter by local linearity with `pulse_detection.filter_zc_linearity` (`LOOKAHEAD_N`, `LINEARITY_CUTOFF`).
2. Group points into pulses with `pulse_detection.greedy_walk` (`TIME_CUTOFF`, `INC_FREQ_CUTOFF_HZ`, `DEC_FREQ_CUTOFF_HZ`, `MIN_POINTS`, `BUFFER_US`).
3. Trim tails with `pulse_cleanup.filter_pulse_tails`.
4. Reject bad pulses with `pulse_cleanup.clean_pulses` (bandwidth, duration, and Fmin/Fmax ranges from `config`). Keep rows where `status > 1`.
5. Compute per-pulse metrics in `pulse_metrics.py` and `calculate_all_metrics`: `fc_freq_hz`, `knee_freq_hz`, `duration_us`, `fmax_hz`, `fmin_hz`, `tstart_us`, `tend_us`.

`process_file` returns `(pulses_df, metrics_df)`. It raises if the file cannot be read or processed. A readable file with no pulses returns empty dataframes.

## Anabat ZC files

`zc/zc_loader.py` ports the loader in `Anabat File Format.pdf` and `C++Loader.txt` (project root). It reads file types 129 to 132.

- Interval bytes become cumulative times in microseconds. The first and second points have no frequency and are dropped.
- Each point's frequency is `divratio * 1e6 / (T[t] - T[t-2])`, which matches the WAV extraction.
- A point is dropped if its status is 0 (out of range) or 1 (switched off), or if its frequency is outside the detector range (the loader's `t_min` and `t_max`). Status 2 (normal) and 3 (main dot) are kept.
- `amplitude` is NaN, because the files hold no amplitude. NaN amplitude passes through PCHIP and normalization unchanged. Training and inference drop samples with NaN in the chosen `FEATURE_COLUMNS`, so do not put `amplitude` in `FEATURE_COLUMNS` when training on ZC files.
- A bad header (unknown type, wrong pointers, `RES1` or division ratio out of range, or a truncated header) raises `ZCFileError`. That is a failed file.
- A file that ends partway through the data keeps the points read before the end. A valid file with no points returns an empty DataFrame, which is a no-pulse file.
- The same cleanup thresholds in `config.py` apply to WAV and ZC files.

## Failed files and no-pulse files

Stage 1 does not catch errors. Each caller decides what to do, and every stage that processes a folder writes two lists:

- Failed: the file could not be read or processed. The list has the path and the exception type and message.
- No pulses: the file was read, but zero pulses survived. The list has the path.

A file where class filtering removes every pulse is not a no-pulse file. Those are counted in the rejection summary.

## Rules for this stage

- Metric functions are wrapped by `run_metric`. It retries per pulse when a whole-file call fails. Keep this fallback when you change a metric function, so one bad pulse does not lose the whole file.
- `calculate_fc` and `calculate_knee` need unique row labels. `calculate_all_metrics` resets the index before calling them. Keep that reset.
- `calculate_fc` and `calculate_knee` return row labels (`fc_idx`, `knee_idx`), not frequencies. Every caller must look up `freq_hz` at those labels. `calculate_all_metrics` does this and produces `fc_freq_hz` and `knee_freq_hz`; `_optional_get_metrics.py` does it for `fc` and `knee`. Never write the labels to an output file as if they were frequencies.
- Keep the int16 scaling step in `load_wav_points`. Removing it changes the amplitude and frequency behavior of the extraction thresholds.
- Metric thresholds for pulse acceptance live in `config.py`, not in the module code.
- Both loaders must return the columns `time_us`, `freq_hz`, `amplitude`, sorted by time, with a default integer index.
- `classifier/inference.py`, `preprocess/pipeline.py`, and `preprocess/get_normalization_stats.py` call `process_file`. Keep its name and its `(pulses_df, metrics_df)` return value.

## Known gaps (verify before relying on them)

- `zc/pipeline.py::main()` only returns dictionaries in memory. It writes no files. It prints each failed file with its reason and leaves failed files out of the results. No-pulse files stay in the results with empty dataframes. Pulse CSVs come from `preprocess/pipeline.py`.
- `zc/output.py::write_pulses` and `config.OUTPUT_MODE` ("individual" or "combined") belong to the `_optional_*` scripts. Those scripts import sibling modules without the package prefix, so they run only from inside `zc/`. Do not treat them as part of the main pipeline unless the user says so.
- `zc/_optional_convert.py` and `zc/_optional_get_metrics.py` read WAV and ZC files. Each has its own copy of the stage 1 steps with per-step timing, and a `load_points(path, times)` function that picks the WAV or ZC loader. Keep them in step with `zc/pipeline.py` when the stage 1 steps change. A file that raises is printed with its reason and left out of the totals; the run continues. They do not write failed-file or no-pulse CSVs.
- `write_pulses` names its CSVs from the full input file name: `recording.wav.csv` and `recording.wav_pulse_1.csv` (combined and individual modes). In combined mode a no-pulse file gets an empty CSV with the header only.
- `config.METRIC_OUTPUT_FILE` is set, but nothing in the main pipeline writes to it. `zc/_optional_get_metrics.py` writes it, as one CSV for all WAV and ZC files under `config.INPUT_DIR`. Columns: `file`, `pulse_id`, `duration`, `fmax`, `fmin`, `fc`, `knee`, `tstart`, `tend`. `duration`, `tstart`, `tend` are in microseconds. `fmax`, `fmin`, `fc`, `knee` are in Hz. Unlike the main pipeline, this script does not use `run_metric`, so an exception inside a metric calculation loses that whole file.
