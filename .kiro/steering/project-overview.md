# BatNET LSTM: Project Overview

This project turns raw bat echolocation WAV recordings into a labeled dataset and trains an LSTM classifier on it.

## Pipeline

Data flows through four stages in this order. Each stage reads the output of the previous one.

| Stage | Folder | Status | Entry point | Output |
|---|---|---|---|---|
| 1. Zero-crossing (ZC) extraction | `zc/` | Built | `python zc/pipeline.py` (the `main()` function returns in-memory results only; see the zc steering file) | Pulse point data per WAV |
| 2. Preprocessing | `preprocess/` | Built | `python preprocess/pipeline.py` | Normalized per-pulse CSVs in `config.OUTPUT_DIR`, plus `pulse_rejection_summary.csv` |
| 3. Dataset creation | `dataset_creation/` | Built | `python dataset_creation/build_shards.py` | Resumable NumPy shards, `class_mapping.json`, `metadata.jsonl`, `manifest.jsonl` in `config.SHARD_PARENT_DIR`, plus `pulse_rejection_summary.csv` |
| 4. Classifier | `classifier/` | Built | `python classifier/train.py`, `python classifier/inference.py`, `python classifier/validation_inference.py` | Run folder with `training_log.txt` and `best_model.pt` in `config.CLASSIFIER_OUTPUT_DIR`; predictions and failed-files CSVs for the test set and the validation set |

Stage details live in their own steering files:
- `zc-stage.md` for `zc/`
- `preprocess-stage.md` for `preprocess/`
- `dataset-creation.md` for `dataset_creation/`
- `classifier.md` for `classifier/`
- `data-contracts.md` for the file formats passed between stages

## Configuration

All tunable parameters and absolute paths live in `config.py` at the project root. Each stage imports `config` and reads from it.

- Do not hardcode paths, thresholds, or feature lists in stage code. Add them to `config.py`.
- Paths in `config.py` are Windows absolute paths under `C:\Users\Students\...`. Use `pathlib.Path` and raw strings when editing them.
- `NUM_WORKERS` controls the process pool size in every stage.
- Changing any parameter that affects features (ZC settings, PCHIP interval, feature columns, normalization) invalidates existing outputs. Regenerate downstream stages after such a change.

## Running the pipeline

Run every script from the project root so the `sys.path` setup in each module resolves `config` and the `preprocess` package correctly.

Recommended order for a full run:

1. `preprocess/get_normalization_stats.py` (only when the feature set or the training data changes; it writes `preprocess/normalization_stats.json`)
2. `preprocess/pipeline.py`
3. `dataset_creation/build_shards.py` on the training WAVs with `VAL_RATIO > 0`, writing to `TRAIN_SHARD_DIR`
4. `classifier/train.py`
5. `classifier/inference.py` on the WAVs in `TEST_DIR`, after pointing `CHECKPOINT_PATH` at a run's `best_model.pt`. It runs stages 1 and 2 internally, without class filtering, and writes no shards.
6. Optional: `classifier/validation_inference.py`, which runs the same pipeline on the validation WAVs. `INPUT_DIR` must still point at the training WAVs.

Run long jobs in your own terminal. They use multiprocessing and can take hours on the full dataset. Do not start them through the agent shell.

## Conventions

- Python with NumPy, pandas, librosa, and tqdm. Use `ProcessPoolExecutor` with an `initializer` for per-worker setup, as the existing stages do.
- Guard every script entry point with `if __name__ == "__main__":`.
- Keep stage modules importable without side effects. Work happens inside functions.
- Class labels come from the immediate parent folder name of each WAV file, for example `.../INPUT_DIR/MYTCAL/recording.wav` has class `MYTCAL`. This rule is used by preprocessing, normalization stats, dataset creation, and inference (`ground_truth`). Keep it consistent.
- The existing code skips failed files silently. When you add error handling, log the file path and the reason. Do not add new silent `except: pass` blocks.

## Current scope

- Do not change pipeline behavior or refactor stages unless the user asks.
- Keep modules short and simple. Do not overcomment or overengineer.
- Do not add tests unless the user asks for them.
