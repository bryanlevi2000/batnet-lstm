# Investigation changes: training collapse

Temporary changes made to investigate the late-training collapse (train and val metrics drop suddenly and don't recover). Undo all of them once the bug is fixed, except any change we decide to keep as the fix. The user keeps a backup of the pre-investigation code.

Every changed line in `classifier/train.py` is marked with `# INVESTIGATION`.

## Change log

| # | Date | File | Change | How to undo |
|---|---|---|---|---|
| 1 | 2026-11-10 | `classifier/diagnostics.py` | New module. `Diagnostics` writes `step_diagnostics.csv` and `epoch_diagnostics.csv` to the run folder. `attention_stats` repeats the `PulseLSTM.forward` math in eval mode to summarize attention weights. | Delete the file. |
| 2 | 2026-11-10 | `classifier/train.py` | Imports `random`, `time`, `numpy`, and `Diagnostics`. | Remove the marked imports. |
| 3 | 2026-11-10 | `classifier/train.py` | `LOGGED_SETTINGS` also logs `SEED`, `GRAD_CLIP_NORM`, `DIAG_LOG_EVERY`, `SAVE_EVERY_EPOCH`, `RESUME_FROM`. | Remove the marked line. |
| 4 | 2026-11-10 | `classifier/train.py` | Seeding: new `seed_epoch()`. Seeds `random`, NumPy, torch, and the train DataLoader generator with `SEED + epoch` at the start of each epoch (and `SEED + 0` before model init). Sets `cudnn.deterministic = True`, `cudnn.benchmark = False`. The train DataLoader gets `generator=`. | Remove `seed_epoch`, the seeding block in `main`, the per-epoch call, and the `generator=` argument. |
| 5 | 2026-11-10 | `classifier/train.py` | `run_epoch` computes the total gradient norm every training step (`torch.nn.utils.get_total_norm`), or clips with `clip_grad_norm_` when `GRAD_CLIP_NORM` is set. It calls `diag.start_step / before_update / after_update` around `optimizer.step()`. New keyword args `diag`, `epoch`, `class_counts`. | Remove the marked block and the extra args. |
| 6 | 2026-11-10 | `classifier/train.py` | Gradient clipping experiment (approved by the user). Off when `GRAD_CLIP_NORM = None`. | Part of #5. Keep only if clipping becomes the fix. |
| 7 | 2026-11-10 | `classifier/train.py` | Per-class validation accuracy collected through `class_counts` and written to `epoch_diagnostics.csv` with the epoch time. | Remove the marked lines. |
| 8 | 2026-11-10 | `classifier/train.py` | Per-epoch checkpoint `run_dir/epochs/epoch_XXX.pt`: model and optimizer state, `step`, `best_loss`, `epochs_without_improvement`, plus the `best_model.pt` fields. The early-stopping check moved after this save, and the log lines are unchanged. | Remove the marked save block and move the early-stopping check back into the `else` branch. |
| 9 | 2026-11-10 | `classifier/train.py` | Resume: when `RESUME_FROM` is set, loads model and optimizer state, overrides LR and weight decay with the current config, and continues from the next epoch with the saved step, best loss, and patience count. Writes a `resumed after epoch ...` line to the log. Always starts a new run folder. | Remove the marked resume block and the log line. |
| 10 | 2026-11-10 | `config.py` | New block under Stage 4: `SEED = 42`, `GRAD_CLIP_NORM = None`, `DIAG_LOG_EVERY = 100`, `SAVE_EVERY_EPOCH = True`, `RESUME_FROM = None`. | Delete the `INVESTIGATION` block. |
| 11 | 2026-11-10 | `classifier/replay_epoch.py` | New script. Starts from `epochs/epoch_<N-1>.pt` of a seeded run and repeats epoch N step by step, exactly as the run did. Writes `replay_epoch_<N>/steps.csv` (every step), `spike_samples.csv` (per-sample gradient and input details for each batch whose gradient norm is above `--spike-threshold`), and `spike_batch_<step>.pt`. At the end it checks the replay against the run's `step_diagnostics.csv`. It writes nothing else. | Delete the file and any `replay_epoch_*` folders in run folders. |

The learning rate scheduler (below) also touches the investigation code in two places:

| # | Date | File | Change | How to undo |
|---|---|---|---|---|
| 12 | 2026-11-11 | `classifier/train.py` | Per-epoch checkpoints also store `scheduler_state` (None when `LR_SCHEDULER = "none"`). On resume, a checkpoint with scheduler state continues its schedule: it keeps the checkpoint's reduced learning rate and plateau counters, and takes `LR_FACTOR`, `LR_PATIENCE`, `LR_MIN` from the current config. A checkpoint without scheduler state resets the learning rate to `LEARNING_RATE` and starts the scheduler fresh. | Remove the `scheduler_state` line and the `continue_schedule` lines in the marked resume block. |
| 13 | 2026-11-11 | `classifier/diagnostics.py`, `classifier/train.py` | `lr` column in `step_diagnostics.csv` (after `step`) and `epoch_diagnostics.csv` (after `end_step`). `train.py` sets `diag.lr` at the start of each epoch. | Goes away with `diagnostics.py`; remove the marked `diag.lr` line. |

## Permanent change (not part of the investigation; do not undo)

Learning rate scheduler, added 2026-11-11 at the user's request. Off by default (`LR_SCHEDULER = "none"`), so existing behavior is unchanged until it is turned on.

- `config.py`: `LR_SCHEDULER`, `LR_FACTOR`, `LR_PATIENCE`, `LR_MIN`, in the Stage 4 section above the `INVESTIGATION` block.
- `classifier/train.py`: `make_scheduler`, the settings checks at the top of `main`, one scheduler step per epoch after validation with a `learning rate reduced to X` log line, and the four settings in `LOGGED_SETTINGS`. These lines carry no `# INVESTIGATION` marker.
- `.kiro/steering/classifier.md`: Training section and Rules section.

Not changed: `model.py`, `dataset.py`, `inference.py`, `validation_inference.py`, and every stage 1 to 3 file. Read-only check scripts and their output are outside the repo in `%TEMP%\batnet_checks\`.

## Output files per run

- `training_log.txt`: as before, plus the new settings.
- `step_diagnostics.csv`: one row every `DIAG_LOG_EVERY` optimizer steps. The `lr` column is the learning rate in use that epoch.
  - Over the interval: `loss_mean`, `loss_max`, `acc`, `grad_norm_mean`, `grad_norm_max`, `clipped_frac`, `nonfinite_steps`.
  - At the logged step: gradient norm per layer group, `param_norm`, `update_ratio` (‖Δθ‖ / ‖θ‖), `update_max_abs`, largest |logit|, median top logit, and attention summaries (median and p99 of the largest attention weight, median effective steps / length).
- `epoch_diagnostics.csv`: epoch losses and accuracies, seconds, the global step at the epoch's end, the learning rate used (`lr`), and validation accuracy per class.
- `epochs/epoch_XXX.pt`: about 4 MB each. Set `RESUME_FROM` to one of these to branch an experiment.

## Experiments

| Run | Change from baseline | Result |
|---|---|---|
| Baseline `run_20261110_172200` | None (`SEED = 42`, no clipping) | No collapse. Early stop at epoch 48, best val loss 0.0454 (epoch 28). Isolated gradient spikes from epoch 21 on, the largest 38.2 at step ~318,900 in epoch 39 (about 300× the typical 0.12), followed by a short dip in train accuracy that recovered. Replay of epoch 39 (exact match with the run): the 38.2 spike at step 318,837 comes from one sample (`LAUINT/00476_NA_LAUINT.wav`, pulse 30), whose own gradient norm is 38.15 against a batch median of 0.010. All of it is in LSTM layer 0 (70% in `weight_hh_l0`); layer 1, LayerNorm, attention and heads are normal. The loss for that sample is ordinary (0.073) and it is classified correctly, so it is not a bad label or a corrupt input. The two next-largest spikes in the epoch (1.16, 1.02) are also single samples. |
| Seed 1, unclipped `run_20261110_210241` | `SEED = 1`, no clipping | Crashed. Epoch 49 was normal (train loss 0.0537, 83.0%). Epoch 50 collapsed (0.0773, 67.4%) and stayed near 0.077 until early stop at epoch 57. Best val loss 0.0455 (epoch 37). |
| Resume from epoch 49, unclipped `run_20261111_000432` | Same settings as above, `RESUME_FROM = epoch_049.pt` | Reproduced the crash exactly (epoch 50: 0.0773, 67.4%). Gradient norm per 100 steps went from about 0.3 to 6.9e4, 2.8e6, 1.5e8 in the interval ending step 404,800 and after, then stayed between 1e9 and 1e16. |
| Resume from epoch 49, clipped `run_20261111_001202` | Same, `GRAD_CLIP_NORM = 1.0` | No crash. Pre-clip gradient norms still reached 4e6 early in epoch 50 and 1e13 later, but each update was capped. Train loss went back to 0.054 within a few intervals and kept slowly improving. Early stop at epoch 57, best val loss 0.0455 (same as unclipped baseline), so clipping fixed the stability but not the accuracy ceiling. |
