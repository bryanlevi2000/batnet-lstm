# evaluate.py
import os

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

import config

from dataset import (
    PulseDataset,
    collate_fn,
    discover_wav_files,
)

from model import LSTMClassifier


# ============================================================
# DEVICE
# ============================================================

torch.set_num_threads(config.NUM_WORKERS)
torch.set_num_interop_threads(config.NUM_WORKERS)


# ============================================================
# EVALUATE
# ============================================================

def evaluate():

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        "best_model.pt",
        map_location=device
    )

    class_names = checkpoint["class_names"]
    class_to_idx = checkpoint["class_to_idx"]

    num_classes = len(class_names)

    print(
        f"Loaded model with {num_classes} classes."
    )

    # --------------------------------------------------------
    # Find test WAV files
    # --------------------------------------------------------

    test_samples, _ = discover_wav_files(
        config.TEST_DIR,
        class_to_idx=class_to_idx
    )

    print(
        f"Found {len(test_samples):,} test WAV files."
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = PulseDataset(
        samples=test_samples,
        normalization_path=config.STATS_FILE
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = LSTMClassifier(
        num_classes=num_classes
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    all_labels = []
    all_predictions = []
    all_confidences = []
    all_files = []

    all_probabilities = []
    all_logits = []

    with torch.no_grad():

        for sequences, labels, lengths, file_paths in test_loader:

            sequences = sequences.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            # ------------------------------------------------
            # Model prediction
            # ------------------------------------------------

            _, logits = model(
                sequences,
                lengths
            )

            # ------------------------------------------------
            # Independent one-vs-rest probabilities
            # ------------------------------------------------

            probabilities = torch.sigmoid(
                logits
            )

            # ------------------------------------------------
            # Top-1 prediction
            # ------------------------------------------------

            confidences, predictions = torch.max(
                probabilities,
                dim=1
            )

            # ------------------------------------------------
            # Store results
            # ------------------------------------------------

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_logits.extend(
                logits.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_confidences.extend(
                confidences.cpu().numpy()
            )

            all_files.extend(
                file_paths
            )

    # --------------------------------------------------------
    # Convert results to arrays
    # --------------------------------------------------------

    prob_array = np.array(
        all_probabilities
    )

    logit_array = np.array(
        all_logits
    )

    # --------------------------------------------------------
    # Create output DataFrame
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "file": [
                os.path.basename(x)
                for x in all_files
            ],

            "ground_truth": [
                class_names[i]
                for i in all_labels
            ],

            "prediction": [
                class_names[i]
                for i in all_predictions
            ],

            "confidence": all_confidences
        }
    )

    # --------------------------------------------------------
    # Add independent class scores
    # --------------------------------------------------------

    for idx, class_name in enumerate(class_names):

        prediction_df[class_name] = (
            prob_array[:, idx]
        )

    # --------------------------------------------------------
    # Add raw logits
    # --------------------------------------------------------

    for idx, class_name in enumerate(class_names):

        prediction_df[
            f"{class_name}_logit"
        ] = logit_array[:, idx]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    prediction_df.to_csv(
        "predictions.csv",
        index=False
    )

    print(
        f"\nSaved {len(prediction_df):,} predictions "
        f"to predictions.csv"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    evaluate()
    