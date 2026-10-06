# train.py
from pathlib import Path
import csv

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import config

from dataset import (
    PulseDataset,
    discover_wav_files,
    split_dataset_samples,
    iter_batches
)

from model import LSTMClassifier

from preprocess import fit_training_normalizer


# ============================================================
# BINARY TARGETS
# ============================================================

def create_binary_targets(labels, num_classes):

    targets = torch.zeros(
        labels.size(0),
        num_classes,
        device=labels.device,
        dtype=torch.float32
    )

    targets.scatter_(
        1,
        labels.unsqueeze(1),
        1.0
    )

    return targets


# ============================================================
# LOSS
# ============================================================

def compute_loss(
    step_logits,
    final_logits,
    labels,
    lengths,
    criterion
):

    num_classes = final_logits.size(1)

    binary_targets = create_binary_targets(
        labels,
        num_classes
    )

    # --------------------------------------------------------
    # Final loss
    # --------------------------------------------------------

    final_loss = criterion(
        final_logits,
        binary_targets
    )

    # --------------------------------------------------------
    # Target replication loss
    # --------------------------------------------------------

    step_loss = torch.zeros(
        (),
        device=step_logits.device,
        dtype=step_logits.dtype
    )

    num_valid_predictions = 0

    max_len = step_logits.size(1)

    for t in range(max_len):

        valid = lengths > t

        if valid.sum() == 0:
            continue

        timestep_loss = criterion(
            step_logits[valid, t, :],
            binary_targets[valid]
        )

        valid_count = valid.sum().item()

        step_loss += (
            timestep_loss * valid_count
        )

        num_valid_predictions += valid_count

    if num_valid_predictions > 0:
        step_loss /= num_valid_predictions

    # --------------------------------------------------------
    # Combined loss
    # --------------------------------------------------------

    loss = (
        config.TARGET_REPLICATION_ALPHA * step_loss
        +
        (1.0 - config.TARGET_REPLICATION_ALPHA) * final_loss
    )

    return loss


# ============================================================
# VALIDATION
# ============================================================

def evaluate_model(
    model,
    data_loader,
    device,
    criterion
):

    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    with torch.no_grad():

        for sequences, labels, lengths, _ in iter_batches(
            data_loader,
            batch_size=config.BATCH_SIZE,
            shuffle=False
        ):

            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            step_logits, final_logits = model(
                sequences,
                lengths
            )

            loss = compute_loss(
                step_logits,
                final_logits,
                labels,
                lengths,
                criterion
            )

            batch_size = labels.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            predictions = torch.argmax(
                final_logits,
                dim=1
            )

            running_correct += (
                predictions == labels
            ).sum().item()

            running_total += batch_size

    if running_total == 0:
        return float("inf"), 0.0

    return (
        running_loss / running_total,
        running_correct / running_total
    )


# ============================================================
# WRITE VALIDATION FILES CSV
# ============================================================

def write_validation_files(validation_samples):

    output_path = (
        Path(__file__).resolve().parent
        / "validation_files.csv"
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "file_path",
            "class_name"
        ])

        for file_path, _ in validation_samples:

            file_path = Path(file_path).resolve()

            writer.writerow([
                str(file_path),
                file_path.parent.name
            ])

    print(
        f"Saved validation file list to:\n"
        f"{output_path}"
    )

# ============================================================
# TRAIN
# ============================================================

def train():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    torch.manual_seed(
        config.RANDOM_SEED
    )

    # ========================================================
    # DISCOVER WAV FILES
    # ========================================================

    all_samples, class_to_idx = discover_wav_files(
        config.TRAIN_DIR
    )

    class_names = [
        class_name
        for class_name, _ in sorted(
            class_to_idx.items(),
            key=lambda x: x[1]
        )
    ]

    print(
        f"Found {len(all_samples):,} WAV files "
        f"across {len(class_names)} classes."
    )

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    train_samples, validation_samples = (
        split_dataset_samples(
            all_samples,
            validation_ratio=config.VALIDATION_RATIO,
            random_seed=config.RANDOM_SEED
        )
    )

    print(
        f"Training WAV files:   {len(train_samples):,}"
    )

    print(
        f"Validation WAV files: {len(validation_samples):,}"
    )

    # ========================================================
    # SAVE VALIDATION FILE LIST
    # ========================================================

    write_validation_files(
        validation_samples
    )

    # ========================================================
    # FIT NORMALIZATION ON TRAINING DATA ONLY
    # ========================================================

    print("\nCalculating training normalization statistics...")

    training_wav_paths = [
        path
        for path, _ in train_samples
    ]

    fit_training_normalizer(
        training_wav_paths
    )

    print(
        f"Saved normalization statistics to:\n"
        f"{config.STATS_FILE}"
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = PulseDataset(
        samples=train_samples,
        normalization_path=config.STATS_FILE
    )

    validation_dataset = None

    if validation_samples:

        validation_dataset = PulseDataset(
            samples=validation_samples,
            normalization_path=config.STATS_FILE
        )

    # ========================================================
    # DATALOADERS
    # ========================================================

    loader_kwargs = {
        "batch_size": None,
        "num_workers": config.NUM_WORKERS,
        "pin_memory": True,
    }

    if config.NUM_WORKERS > 0:

        loader_kwargs["persistent_workers"] = True

        # Controls how many WAV results each worker can have
        # prepared ahead of time.
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        **loader_kwargs
    )

    validation_loader = None

    if validation_dataset is not None:

        validation_loader = DataLoader(
            validation_dataset,
            **loader_kwargs
        )

    # ========================================================
    # MODEL
    # ========================================================

    num_classes = len(class_names)

    print(f"\nClasses: {num_classes}")
    print(f"Features: {len(config.FEATURE_COLUMNS)}")
    print(
        f"Feature columns: "
        f"{config.FEATURE_COLUMNS}"
    )

    model = LSTMClassifier(
        num_classes=num_classes
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.NUM_EPOCHS):

        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for sequences, labels, lengths, _ in iter_batches(
            train_loader,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            shuffle_buffer_size=config.SHUFFLE_BUFFER_SIZE,
            random_seed=config.RANDOM_SEED,
            epoch=epoch
        ):

            sequences = sequences.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            lengths = lengths.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            step_logits, final_logits = model(
                sequences,
                lengths
            )

            loss = compute_loss(
                step_logits,
                final_logits,
                labels,
                lengths,
                criterion
            )

            loss.backward()

            optimizer.step()

            batch_size = labels.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            predictions = torch.argmax(
                final_logits,
                dim=1
            )

            running_correct += (
                predictions == labels
            ).sum().item()

            running_total += batch_size

        if running_total == 0:
            raise RuntimeError(
                "No training samples were produced."
            )

        epoch_loss = (
            running_loss / running_total
        )

        epoch_accuracy = (
            running_correct / running_total
        )

        print(
            f"\nEpoch "
            f"{epoch + 1:03d}/{config.NUM_EPOCHS} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Accuracy: {epoch_accuracy:.4f}"
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        monitor_loss = epoch_loss
        monitor_accuracy = epoch_accuracy

        if validation_loader is not None:

            validation_loss, validation_accuracy = (
                evaluate_model(
                    model,
                    validation_loader,
                    device,
                    criterion
                )
            )

            monitor_loss = validation_loss
            monitor_accuracy = validation_accuracy

            print(
                f"Validation Loss: "
                f"{validation_loss:.4f} | "
                f"Validation Accuracy: "
                f"{validation_accuracy:.4f}"
            )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if monitor_loss < best_loss - 1e-6:

            best_loss = monitor_loss
            patience_counter = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "class_to_idx": class_to_idx,
                "epoch": epoch + 1,
                "loss": monitor_loss,
                "normalization_path": str(
                    config.STATS_FILE
                ),
            }

            torch.save(
                checkpoint,
                "best_model.pt"
            )

            print(
                f"Saved best model "
                f"(loss={monitor_loss:.4f})"
            )

        else:

            patience_counter += 1

            print(
                f"No improvement for "
                f"{patience_counter}/"
                f"{config.PATIENCE} epochs."
            )

            if (
                config.PATIENCE > 0
                and patience_counter >= config.PATIENCE
            ):

                print(
                    "Early stopping triggered."
                )

                break

    print("\nTraining complete.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    train()
