"""Train the classifier. Run from the project root: python classifier/train.py"""

import json
import random  # INVESTIGATION
import sys
import time  # INVESTIGATION
from datetime import datetime
from pathlib import Path

import numpy as np  # INVESTIGATION
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from dataset import PulseDataset, collate_fn, load_split
from diagnostics import Diagnostics  # INVESTIGATION
from model import PulseLSTM


LOGGED_SETTINGS = [
    "TRAIN_SHARD_DIR", "FEATURE_COLUMNS", "BATCH_SIZE", "HIDDEN_SIZE", "NUM_LAYERS",
    "LEARNING_RATE", "DROPOUT", "ATTENTION_DROPOUT", "TARGET_REPLICATION_ALPHA",
    "NUM_EPOCHS", "MAX_SEQUENCE_LENGTH", "PATIENCE", "WEIGHT_DECAY",
    "LR_SCHEDULER", "LR_FACTOR", "LR_PATIENCE", "LR_MIN",
    "SEED", "GRAD_CLIP_NORM", "DIAG_LOG_EVERY", "SAVE_EVERY_EPOCH", "RESUME_FROM",  # INVESTIGATION
]


# INVESTIGATION: seeding. Reseeded at the start of every epoch so that a run resumed
# from an epoch checkpoint sees the same shuffle order and dropout masks as the original.
def seed_epoch(seed, epoch, generator):
    random.seed(seed + epoch)
    np.random.seed(seed + epoch)
    torch.manual_seed(seed + epoch)
    generator.manual_seed(seed + epoch)


def make_scheduler(optimizer):
    """Learning rate scheduler from the config, or None. Stepped once per epoch with the validation loss."""

    if config.LR_SCHEDULER == "none":
        return None

    # threshold=0 makes "better" mean a strictly lower loss, the same test early stopping uses.
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.LR_FACTOR, patience=config.LR_PATIENCE,
        threshold=0.0, min_lr=config.LR_MIN,
    )


def step_loss(step_logits, y, lengths):
    """BCE at every real time step, averaged per sample, then over the batch."""

    loss = F.binary_cross_entropy_with_logits(
        step_logits, y[:, None, :].expand_as(step_logits), reduction="none"
    ).mean(dim=2)

    lengths = lengths.to(loss.device)
    mask = torch.arange(loss.size(1), device=loss.device)[None, :] < lengths[:, None]

    return (loss.masked_fill(~mask, 0).sum(dim=1) / lengths).mean()


def run_epoch(model, loader, device, n_classes, alpha, optimizer=None,
              diag=None, epoch=None, class_counts=None):  # INVESTIGATION: diag, epoch, class_counts
    """Train when an optimizer is given, otherwise validate. Returns (loss, accuracy %)."""

    training = optimizer is not None
    model.train(training)
    total_loss = correct = count = 0

    with torch.set_grad_enabled(training):

        for x, lengths, labels in loader:

            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            y = F.one_hot(labels, n_classes).float()

            if training and alpha > 0:
                pooled_logits, step_logits = model(x, lengths, return_steps=True)
                loss = (
                    alpha * step_loss(step_logits, y, lengths)
                    + (1 - alpha) * F.binary_cross_entropy_with_logits(pooled_logits, y)
                )
            else:
                pooled_logits = model(x, lengths)
                loss = F.binary_cross_entropy_with_logits(pooled_logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()

                # INVESTIGATION: gradient norm (and optional clipping) plus step diagnostics.
                params = [p for p in model.parameters() if p.grad is not None]
                if config.GRAD_CLIP_NORM:
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP_NORM)
                else:
                    grad_norm = torch.nn.utils.get_total_norm([p.grad for p in params])
                log_now = diag is not None and diag.start_step()
                if log_now:
                    diag.before_update(model)

                optimizer.step()

                if diag is not None:  # INVESTIGATION
                    diag.after_update(
                        model, epoch, loss, grad_norm, pooled_logits, labels, x, lengths, log_now
                    )

            total_loss += loss.item() * len(labels)
            correct += (pooled_logits.argmax(dim=1) == labels).sum().item()
            count += len(labels)

            if class_counts is not None:  # INVESTIGATION: per-class validation accuracy
                hits = labels[pooled_logits.argmax(dim=1) == labels]
                class_counts[:, 0] += torch.bincount(hits, minlength=n_classes).cpu()
                class_counts[:, 1] += torch.bincount(labels, minlength=n_classes).cpu()

    return total_loss / count, 100 * correct / count


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA, but no CUDA device is available.")

    alpha = config.TARGET_REPLICATION_ALPHA

    if not 0 <= alpha <= 1:
        raise ValueError(f"TARGET_REPLICATION_ALPHA must be between 0 and 1, got {alpha}.")

    if config.LR_SCHEDULER not in ("none", "plateau"):
        raise ValueError(f"LR_SCHEDULER must be 'none' or 'plateau', got {config.LR_SCHEDULER!r}.")

    if config.LR_SCHEDULER == "plateau" and not 0 < config.LR_FACTOR < 1:
        raise ValueError(f"LR_FACTOR must be between 0 and 1, got {config.LR_FACTOR}.")

    # INVESTIGATION: seeding and deterministic cuDNN.
    generator = None
    if config.SEED is not None:
        seed_epoch(config.SEED, 0, torch.Generator())
        generator = torch.Generator()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda")
    shard_dir = Path(config.TRAIN_SHARD_DIR)

    with open(shard_dir / "class_mapping.json", encoding="utf-8") as file:
        class_mapping = json.load(file)

    # Load both splits into RAM once.
    datasets = {}

    for split in ("train", "validation"):

        features, offsets, labels, _ = load_split(
            shard_dir, split, config.FEATURE_COLUMNS, config.MAX_SEQUENCE_LENGTH
        )

        if len(labels) == 0:
            raise RuntimeError(
                f"The '{split}' split in {shard_dir} has no samples. "
                "Training needs train and validation splits (build shards with VAL_RATIO > 0)."
            )

        datasets[split] = PulseDataset(features, offsets, labels)

    loaders = {
        split: DataLoader(
            data,
            batch_size=config.BATCH_SIZE,
            shuffle=split == "train",
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
            generator=generator if split == "train" else None,  # INVESTIGATION
        )
        for split, data in datasets.items()
    }

    n_classes = len(class_mapping)
    model_args = {
        "input_size": len(config.FEATURE_COLUMNS),
        "hidden_size": config.HIDDEN_SIZE,
        "num_layers": config.NUM_LAYERS,
        "n_classes": n_classes,
        "dropout": config.DROPOUT,
        "attention_dropout": config.ATTENTION_DROPOUT,
    }
    model = PulseLSTM(**model_args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = make_scheduler(optimizer)

    best_loss = float("inf")
    epochs_without_improvement = 0

    # INVESTIGATION: resume from an epoch checkpoint (model + optimizer state).
    start_epoch, start_step = 1, 0
    if config.RESUME_FROM:
        state = torch.load(config.RESUME_FROM, map_location=device)
        if state["model_args"] != model_args or state["class_mapping"] != class_mapping:
            raise ValueError(
                f"RESUME_FROM {config.RESUME_FROM} was trained with different model_args or "
                "class_mapping than the current config."
            )
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])  # includes the checkpoint's current learning rate
        # A checkpoint saved by a scheduled run continues its schedule: the reduced learning rate and
        # the plateau counters are kept, and factor, patience and minimum come from the current config.
        # Otherwise the learning rate is reset to LEARNING_RATE and the scheduler starts fresh.
        continue_schedule = scheduler is not None and state.get("scheduler_state") is not None
        for group in optimizer.param_groups:  # current config wins, so branches can change them
            if not continue_schedule:
                group["lr"] = config.LEARNING_RATE
            group["weight_decay"] = config.WEIGHT_DECAY
        if continue_schedule:
            scheduler.load_state_dict(state["scheduler_state"])
            scheduler.factor, scheduler.patience = config.LR_FACTOR, config.LR_PATIENCE
            scheduler.min_lrs = [config.LR_MIN] * len(optimizer.param_groups)
        start_epoch = state["epoch"] + 1
        start_step = state["step"]
        best_loss = state["best_loss"]
        epochs_without_improvement = state["epochs_without_improvement"]

    run_dir = Path(config.CLASSIFIER_OUTPUT_DIR) / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    log_path = run_dir / "training_log.txt"
    checkpoint_path = run_dir / "best_model.pt"
    epoch_dir = run_dir / "epochs"  # INVESTIGATION
    if config.SAVE_EVERY_EPOCH:
        epoch_dir.mkdir()

    def log(line):
        print(line)
        with open(log_path, "a", encoding="utf-8") as file:
            file.write(line + "\n")

    for name in LOGGED_SETTINGS:
        log(f"{name}: {getattr(config, name)}")

    log(f"classes: {n_classes}")
    log(f"train samples: {len(datasets['train']):,}")
    log(f"validation samples: {len(datasets['validation']):,}")

    if alpha > 0:
        log("note: train loss includes the target replication term; val loss is pooled only")

    if config.RESUME_FROM:  # INVESTIGATION
        log(f"resumed after epoch {start_epoch - 1} (step {start_step}), "
            f"best val loss so far {best_loss:.4f}, patience {epochs_without_improvement}/{config.PATIENCE}")

    log("")

    # INVESTIGATION
    class_names = sorted(class_mapping, key=class_mapping.get)
    diag = Diagnostics(run_dir, config.DIAG_LOG_EVERY, config.GRAD_CLIP_NORM, class_names, start_step)

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):

        if config.SEED is not None:  # INVESTIGATION
            seed_epoch(config.SEED, epoch, generator)
        started = time.time()  # INVESTIGATION
        diag.lr = optimizer.param_groups[0]["lr"]  # INVESTIGATION: learning rate used this epoch
        class_counts = torch.zeros(n_classes, 2, dtype=torch.int64)  # INVESTIGATION

        train_loss, train_acc = run_epoch(
            model, loaders["train"], device, n_classes, alpha, optimizer, diag=diag, epoch=epoch
        )
        val_loss, val_acc = run_epoch(
            model, loaders["validation"], device, n_classes, alpha, class_counts=class_counts
        )

        log(
            f"epoch {epoch}/{config.NUM_EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.1f}% | "
            f"val loss {val_loss:.4f} acc {val_acc:.1f}%"
        )
        diag.write_epoch(  # INVESTIGATION
            epoch, train_loss, train_acc, val_loss, val_acc, time.time() - started, class_counts
        )

        if val_loss < best_loss:

            best_loss = val_loss
            epochs_without_improvement = 0

            torch.save({
                "model_state": model.state_dict(),
                "model_args": model_args,
                "feature_columns": list(config.FEATURE_COLUMNS),
                "max_sequence_length": config.MAX_SEQUENCE_LENGTH,
                "class_mapping": class_mapping,
                "epoch": epoch,
                "val_loss": val_loss,
            }, checkpoint_path)

            log(f"saved model (val loss {val_loss:.4f})")

        else:

            epochs_without_improvement += 1
            log(f"no improvement for {epochs_without_improvement}/{config.PATIENCE} epochs")

        if scheduler is not None:

            lr_before = optimizer.param_groups[0]["lr"]
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]["lr"]

            if lr != lr_before:
                log(f"learning rate reduced to {lr:.2e}")

        # INVESTIGATION: per-epoch checkpoint with optimizer state, usable as RESUME_FROM.
        if config.SAVE_EVERY_EPOCH:
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "model_args": model_args,
                "feature_columns": list(config.FEATURE_COLUMNS),
                "max_sequence_length": config.MAX_SEQUENCE_LENGTH,
                "class_mapping": class_mapping,
                "epoch": epoch,
                "step": diag.step,
                "val_loss": val_loss,
                "best_loss": best_loss,
                "epochs_without_improvement": epochs_without_improvement,
            }, epoch_dir / f"epoch_{epoch:03d}.pt")

        if epochs_without_improvement >= config.PATIENCE:
            log(f"early stopping: no improvement for {config.PATIENCE} epochs")
            break

    diag.close()  # INVESTIGATION
    log(f"best val loss {best_loss:.4f}, checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
