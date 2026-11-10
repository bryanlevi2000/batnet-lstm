---
inclusion: fileMatch
fileMatchPattern: "dataset_creation/**"
---

# Stage 3: Dataset Creation

Folder: `dataset_creation/`. The main script is `build_shards.py`. It runs stages 1 and 2 on every WAV and writes NumPy shards for training.

## What one sample is

Each positive `pulse_id` in a WAV's preprocessed output is one sample. Its features are the `NORMALIZATION_COLUMNS` values, in order, for every time step of that pulse. Its label is the integer for the WAV's class folder.

## Splitting

`find_wavs` splits by WAV file, never by pulse.

- WAVs are grouped by class (parent folder). Within each class they are shuffled with `random.Random(f"{SPLIT_SEED}:{class_name}")`, so the split is stable across runs.
- With `VAL_RATIO > 0`, each class keeps at least one training WAV. Validation gets `round(len * VAL_RATIO)`, capped at `len - 1`.
- All pulses from one WAV stay in one split. Keep it that way, or pulses from the same recording leak between train and validation.

## Shard writing and resume

- `SHARD_NUM_SAMPLES` caps samples per shard. A shard is written when it fills, and the remainder is written at the end of each split.
- Each shard writes three `.npy` files to temp names, appends its metadata lines to `metadata.jsonl`, renames the temp files, then appends one line to `manifest.jsonl`. The manifest line is the commit point.
- On startup, `load_progress` truncates `manifest.jsonl` and `metadata.jsonl` to the last complete entry. It counts how many samples each WAV already contributed and skips them.
- Resume depends on deterministic WAV order (`sorted` and `dirs.sort()`) and unchanged config. If you change the feature set, filters, or `SHARD_NUM_SAMPLES` between runs, resume mixes old and new data. Write to a new `SHARD_PARENT_DIR` instead.
- `class_mapping.json` is loaded and extended on each run. Existing labels are never reassigned. Do not edit the file by hand unless you also rebuild the shards.

## Rejection summary

- `process_split` accumulates each WAV's `rejection_stats` by class, across both splits, into one dict. `main()` writes it with `filtering.save_rejection_summary` to `config.PULSE_FILTERED_REJECTION_FILE` after both splits finish.
- Every run processes every WAV, including on resume, so the summary always covers the full dataset. It is written only at the end, so a crashed run leaves the previous file in place.
- WAVs where stage 1 fails or finds no pulses are not counted. This matches `preprocess/pipeline.py`.

Output format is in `data-contracts.md`.

## Rules for this stage

- Keep the write order: arrays, then metadata, then manifest. Do not change it, since resume depends on it.
- Do not parallelize the writer. Only the WAV processing runs in `ProcessPoolExecutor`.
- Keep `features` float32 and `labels` and `offsets` int64.
- Use `np.save` with `allow_pickle=False`.
