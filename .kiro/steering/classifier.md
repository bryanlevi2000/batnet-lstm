---
inclusion: fileMatch
fileMatchPattern: "classifier/**"
---

# Stage 4: Classifier

Folder: `classifier/`. An LSTM that labels each pulse with a class. All settings are in the "Stage 4: Classifier" section of `config.py`.

## Modules

| File | Contents |
|---|---|
| `dataset.py` | `read_manifest`, `check_ram`, `load_split`, `PulseDataset`, `collate_fn` (training only) |
| `model.py` | `PulseLSTM` |
| `train.py` | Entry point: `python classifier/train.py` |
| `inference.py` | Entry point: `python classifier/inference.py`. Also defines `run(wav_paths, predictions_path, failed_path)` |
| `validation_inference.py` | Entry point: `python classifier/validation_inference.py` |

Run all scripts from the project root. `train.py` does not import from the inference scripts, and they don't import from it. `validation_inference.py` reuses `inference.run`. Only `train.py` uses `dataset.py`.

## Training

- Reads the `train` and `validation` splits from `config.TRAIN_SHARD_DIR`, which must be built with `VAL_RATIO > 0`.
- `load_split` reads shards listed in `manifest.jsonl` and checks RAM against `psutil` available memory minus `RAM_RESERVE_GB`. It copies `FEATURE_COLUMNS` into one float32 array, truncates samples to `MAX_SEQUENCE_LENGTH`, and drops samples with NaN. Column indexes come from `config.NORMALIZATION_COLUMNS`.
- `collate_fn` zero-pads each batch to its longest sample. `PulseLSTM` packs the batch with `pack_padded_sequence`, so the LSTM skips padding.
- `PulseLSTM` applies, in order: a unidirectional LSTM, LayerNorm, dropout, masked attention pooling with attention dropout, then one `Linear(hidden, n_classes)`. Each output is an independent sigmoid head trained with BCE.
- Training loss is `alpha * step_loss + (1 - alpha) * pooled_loss`, where alpha is `TARGET_REPLICATION_ALPHA`. Step loss is masked and divided by each sample's length. With alpha 0, step logits are not computed. Validation and inference use only the pooled output.
- Uses AdamW and shuffles training batches only. Requires CUDA. DataLoaders use `num_workers=0`, because on Windows each worker would copy the in-RAM dataset.
- After each epoch it logs the epoch line, then either `saved model (val loss X)` or `no improvement for k/PATIENCE epochs`. It stops early after `PATIENCE` epochs without improvement.
- Output: `CLASSIFIER_OUTPUT_DIR/run_YYYYMMDD_HHMMSS/` containing `training_log.txt` and `best_model.pt`.

## Inference

- Finds WAVs recursively under `config.TEST_DIR`. Each WAV's parent folder name is its `ground_truth`.
- Each WAV is processed in a worker process (`NUM_WORKERS`): stage 1 ZC extraction (`zc.pipeline._process_wav_unsafe`, including `clean_pulses`), PCHIP resampling, slope and position features, then z-scoring with `preprocess/normalization_stats.json`.
- Class-specific pulse filtering (`preprocess/filtering.py`) is not applied. No shards or intermediate files are written.
- Uses the checkpoint's `feature_columns`, `max_sequence_length`, `model_args`, and `class_mapping`, not the current config. The feature columns must be in `NORMALIZATION_COLUMNS`.
- Results stream back in sorted file order with a bounded number of files in flight. The model runs on the CPU, and rows are appended to `PREDICTIONS_PATH` every `BATCH_SIZE` samples. tqdm shows progress per WAV.
- WAVs that raise an error are written to `FAILED_FILES_PATH` with columns `file` and `error`. A WAV with no pulses is not a failure; it just has no rows.

## Validation inference

- `validation_inference.py` runs the same pipeline as `inference.py` (`inference.run`) on the validation WAVs, so its output is directly comparable to the test predictions.
- The file list is the unique `wav_path` values with `split == "validation"` in `TRAIN_SHARD_DIR/metadata.jsonl`, sorted and joined with `config.INPUT_DIR`. `INPUT_DIR` must still point at the training WAVs.
- Unreadable metadata lines are skipped and counted. The script raises an error if the readable validation lines don't match the manifest's validation sample count.
- `metadata.jsonl` only lists WAVs that kept at least one pulse after class filtering during the shard build. Validation WAVs with no surviving pulses are not included.
- Writes `VAL_PREDICTIONS_PATH` and `VAL_FAILED_FILES_PATH`, in the same formats as `PREDICTIONS_PATH` and `FAILED_FILES_PATH`.

Checkpoint and CSV formats are in `data-contracts.md`.

## Rules for this stage

- Keep the modules short and simple. Do not add modules, class weighting, schedulers, gradient clipping, seeding, or mixed precision unless the user asks.
- Keep inference preprocessing in the same order as `preprocess/pipeline.py::process_wav_file`, minus class filtering.
- Shard inputs are already z-score normalized. Do not normalize them again.
- Keep the WAV-level split as built. Do not re-split pulses.
- Do not start a training run from the agent shell. The user runs training and inference.
