---
inclusion: fileMatch
fileMatchPattern: "{classifier/**,dataset_creation/**,preprocess/**}"
---

# Data Contracts Between Stages

These formats are the interface between stages. A change to any of them must be made in every stage that reads or writes the format.

## Stage 2 output: preprocessed pulse CSVs

Written by `preprocess/pipeline.py` to `config.OUTPUT_DIR`, mirroring the folder structure of `config.INPUT_DIR`.

- One CSV per WAV file, named `<wav_stem>.csv`, in the matching class subfolder.
- One row per pulse point. Columns include `time_us`, `freq_hz`, `amplitude`, `pulse_id`, `status`, `slope`, `position`.
- `freq_hz` and `amplitude` are already z-score normalized using `preprocess/normalization_stats.json`. Do not normalize them again.
- `pulse_id` identifies each pulse within a file. Values greater than 0 are real pulses.
- Rejection counts per class are written to `config.PULSE_FILTERED_REJECTION_FILE`. `dataset_creation/build_shards.py` writes the same file, in the same format, to the same path. Whichever script runs last overwrites it. Columns: `class`, `total_pulses`, `accepted`, `rejected`, `rejected_fmin`, `rejected_fmax`, `rejected_fc`, `rejected_duration`. A pulse can fail more than one filter, so the `rejected_*` columns can add up to more than `rejected`.

## Stage 3 output: NumPy shards

Written by `dataset_creation/build_shards.py` to `config.SHARD_PARENT_DIR`.

Layout:

```
SHARD_PARENT_DIR/
  class_mapping.json          class name -> integer label (persistent across runs)
  metadata.jsonl              one JSON line per sample, in shard order
  manifest.jsonl              one JSON line per completed shard (resume source of truth)
  train/
    train_000000_features.npy
    train_000000_offsets.npy
    train_000000_labels.npy
    ...
  validation/                 only created when config.VAL_RATIO > 0
```

Array contents for one shard:

- `features`: float32, shape `(total_time_steps, n_features)`. All samples in the shard are concatenated along the first axis. `n_features` is `len(config.NORMALIZATION_COLUMNS)`, currently 2, in the order `freq_hz`, `amplitude`.
- `offsets`: int64, shape `(num_samples + 1,)`. Sample `i` is `features[offsets[i]:offsets[i+1]]`. The first value is 0.
- `labels`: int64, shape `(num_samples,)`. Integer class label for each sample.

Sample length varies per pulse. Each sample is resampled onto a fixed time grid (`config.PCHIP_TIME_INTERVAL`, currently 100 us), so its length depends on pulse duration. Any model must handle variable-length sequences, for example with padding plus a mask or with packed sequences.

`metadata.jsonl` fields per sample: `split`, `wav_path` (relative to `INPUT_DIR`, POSIX style), `pulse_id`, `label`, `num_time_steps`, `shard`, `sample_index`. Use `wav_path` to trace a sample back to its recording.

## Stage 4 output: checkpoint

`best_model.pt`, written by `classifier/train.py` to its run folder. It's a dict saved with `torch.save`:

- `model_state`: `PulseLSTM` state dict from the best epoch
- `model_args`: constructor kwargs; rebuild with `PulseLSTM(**model_args)`
- `feature_columns`: `FEATURE_COLUMNS` used for training
- `max_sequence_length`: `MAX_SEQUENCE_LENGTH` used for training
- `class_mapping`: class name to label index, from the training shard dir
- `epoch`: 1-based epoch of the saved model
- `val_loss`: validation loss of that epoch

## Stage 4 output: predictions CSV

Written by `classifier/inference.py` to `config.PREDICTIONS_PATH`. There is one row per pulse found in the WAVs under `config.TEST_DIR`, in sorted file order, then pulse order. Pulses do not go through class-specific filtering. Samples with NaN are dropped. Columns, in order:

1. `file`: full WAV path
2. `pulse_id`
3. `ground_truth`: WAV parent folder name
4. One column per class, named with the class name, holding the sigmoid confidence
5. One `<class>_logit` column per class, holding the raw logit

Both per-class groups are ordered by the checkpoint's label index.

`classifier/validation_inference.py` writes the same format to `config.VAL_PREDICTIONS_PATH` for the validation WAVs listed in `TRAIN_SHARD_DIR/metadata.jsonl`.

## Stage 4 output: failed files CSV

Written by `classifier/inference.py` to `config.FAILED_FILES_PATH`, and by `validation_inference.py` to `config.VAL_FAILED_FILES_PATH`. Columns are `file` (full WAV path) and `error` (exception type and message). It is written even when empty. WAVs with no detected pulses are not listed.

## Normalization statistics

`preprocess/normalization_stats.json` holds the mean and standard deviation for each feature. The file also contains `slope` and `position` entries that are not used in shards today. The features used are the ones listed in `config.NORMALIZATION_COLUMNS`. Keep the stats file and that list in sync.

`classifier/inference.py` reads this file at run time, and the checkpoint doesn't store the stats. Regenerating the file changes the inputs every existing checkpoint sees. Retrain after regenerating it.

## Changing the feature set

Adding or removing a feature requires all of the following:

1. Update `config.NORMALIZATION_COLUMNS`.
2. Regenerate `normalization_stats.json` with `get_normalization_stats.py`.
3. Rerun `preprocess/pipeline.py` and `build_shards.py` into a new output directory. Existing shards have the old width and are not compatible.
4. Set `FEATURE_COLUMNS` to a subset of the new list and retrain. The model input size follows `len(FEATURE_COLUMNS)`.
