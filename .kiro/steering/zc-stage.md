---
inclusion: fileMatch
fileMatchPattern: "zc/**"
---

# Stage 1: Zero-Crossing (ZC) Extraction

Folder: `zc/`. Turns a WAV recording into time-ordered zero-crossing points, then groups them into FM pulses.

## Processing order for one WAV

Implemented in `zc/pipeline.py::_process_wav_unsafe`:

1. Load the WAV with `librosa.load(sr=None, mono=True)`.
2. Denoise with `denoise.denoise_audio` using `config.NOISE_FACTOR` and `config.NOISE_STRIDE`.
3. Convert to int16 (`* 32767`). ZC extraction expects this scale.
4. Extract zero crossings with `zc_extraction.extract_zc` (`config.DIVRAT`, `config.HYSTERESIS_PERCENTILE`). Output columns: `time_us`, `freq_hz`, `amplitude`.
5. Filter by local linearity with `pulse_detection.filter_zc_linearity` (`LOOKAHEAD_N`, `LINEARITY_CUTOFF`).
6. Group points into pulses with `pulse_detection.greedy_walk` (`TIME_CUTOFF`, `INC_FREQ_CUTOFF_HZ`, `DEC_FREQ_CUTOFF_HZ`, `MIN_POINTS`, `BUFFER_US`).
7. Trim tails with `pulse_cleanup.filter_pulse_tails`.
8. Reject bad pulses with `pulse_cleanup.clean_pulses` (bandwidth, duration, and Fmin/Fmax ranges from `config`). Keep rows where `status > 1`.
9. Compute per-pulse metrics in `pulse_metrics.py` and `calculate_all_metrics`: `fc_freq_hz`, `knee_freq_hz`, `duration_us`, `fmax_hz`, `fmin_hz`, `tstart_us`, `tend_us`.

`zc/pipeline.py::process_wav` returns `(pulses_df, metrics_df)` or `None`.

## Rules for this stage

- Metric functions are wrapped by `run_metric`. It retries per pulse when a whole-file call fails. Keep this fallback when you change a metric function, so one bad pulse does not lose the whole file.
- `calculate_fc` and `calculate_knee` need unique row labels. `calculate_all_metrics` resets the index before calling them. Keep that reset.
- Keep the int16 scaling step. Removing it changes the amplitude and frequency behavior of the extraction thresholds.
- Metric thresholds for pulse acceptance live in `config.py`, not in the module code.
- `classifier/inference.py` calls `_process_wav_unsafe` directly, so it can record the reason a file failed. Keep that function's name and its `(pulses_df, metrics_df)` return value.

## Known gaps (verify before relying on them)

- `zc/pipeline.py::main()` only returns dictionaries in memory. It writes no files. Pulse CSVs come from `preprocess/pipeline.py`, which calls `process_wav` directly.
- `zc/output.py::write_pulses` and `config.OUTPUT_MODE` ("individual" or "combined") belong to the `_optional_*` scripts. Those scripts import sibling modules without the package prefix, so they run only from inside `zc/`. Do not treat them as part of the main pipeline unless the user says so.
- `config.METRIC_OUTPUT_FILE` is set, but nothing in the main pipeline writes to it.
