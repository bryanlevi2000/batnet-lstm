"""Train the classifier. Run from the project root: python classifier/train.py"""

import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from dataset import PulseDataset, collate_fn, load_split
from model import PulseLSTM


LOGGED_SETTINGS = [
    "TRAIN_SHARD_DIR", "FEATURE_COLUMNS", "BATCH_SIZE", "HIDDEN_SIZE", "NUM_LAYERS",
    "LEARNING_RATE", "DROPOUT", "ATTENTION_DROPOUT", "TARGET_REPLICATION_ALPHA",
    "NUM_EPOCHS", "MAX_SEQUENCE_LENGTH", "PATIENCE", "WEIGHT_DECAY",
]


def step_loss(step_logits, y, lengths):
    """BCE at every real time step, averaged per sample, then over the batch."""

    loss = F.binary_cross_entropy_with_logits(
        step_logits, y[:, None, :].expand_as(step_logits), reduction="none"
    ).mean(dim=2)

    lengths = lengths.to(loss.device)
    mask = torch.arange(loss.size(1), device=loss.device)[None, :] < lengths[:, None]

    return (loss.masked_fill(~mask, 0).sum(dim=1) / lengths).mean()


def run_epoch(model, loader, device, n_classes, alpha, optimizer=None):
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
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (pooled_logits.argmax(dim=1) == labels).sum().item()
            count += len(labels)

    return total_loss / count, 100 * correct / count


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA, but no CUDA device is available.")

    alpha = config.TARGET_REPLICATION_ALPHA

    if not 0 <= alpha <= 1:
        raise ValueError(f"TARGET_REPLICATION_ALPHA must be between 0 and 1, got {alpha}.")

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

    run_dir = Path(config.CLASSIFIER_OUTPUT_DIR) / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    log_path = run_dir / "training_log.txt"
    checkpoint_path = run_dir / "best_model.pt"

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

    log("")

    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):

        train_loss, train_acc = run_epoch(
            model, loaders["train"], device, n_classes, alpha, optimizer
        )
        val_loss, val_acc = run_epoch(model, loaders["validation"], device, n_classes, alpha)

        log(
            f"epoch {epoch}/{config.NUM_EPOCHS} | "
            f"train loss {train_loss:.4f} acc {train_acc:.1f}% | "
            f"val loss {val_loss:.4f} acc {val_acc:.1f}%"
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

            if epochs_without_improvement >= config.PATIENCE:
                log(f"early stopping: no improvement for {config.PATIENCE} epochs")
                break

    log(f"best val loss {best_loss:.4f}, checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
