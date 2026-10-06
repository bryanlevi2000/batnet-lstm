# evaluate.py

import os
import threading
import queue
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import config
from dataset import discover_wav_files
from preprocess import preprocess_wav
from model import LSTMClassifier


# ============================================================
# CPU / WORKER CONFIGURATION
# ============================================================

TOTAL_CORES = os.cpu_count() or 1

if os.name == "nt":
    NUM_WORKERS = min(
        config.NUM_WORKERS,
        max(4, TOTAL_CORES // 4),
        8,
    )
else:
    NUM_WORKERS = min(
        config.NUM_WORKERS,
        max(4, TOTAL_CORES // 2),
    )

MODEL_THREADS = max(
    2,
    min(8, TOTAL_CORES // max(1, NUM_WORKERS))
)

torch.set_num_threads(MODEL_THREADS)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ============================================================
# OPTIONAL THREADPOOL LIMITING
# ============================================================

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

_worker_threadpool = None


def worker_init(worker_id):
    global _worker_threadpool

    torch.set_num_threads(1)

    if threadpool_limits is not None:
        _worker_threadpool = threadpool_limits(limits=1)
        _worker_threadpool.__enter__()


# ============================================================
# WAV-LEVEL EVALUATION DATASET
# ============================================================

class EvaluationWavDataset(IterableDataset):

    def __init__(self, samples, normalization_path):
        super().__init__()
        self.samples = list(samples)
        self.normalization_path = normalization_path

    def __iter__(self):

        worker_info = get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        worker_samples = self.samples[
            worker_id::num_workers
        ]

        for wav_path, label in worker_samples:

            pulses = preprocess_wav(
                wav_path,
                normalization_path=self.normalization_path
            )

            valid_pulses = []

            for sequence in pulses:

                if sequence.numel() == 0:
                    continue

                if len(sequence) > config.MAX_SEQUENCE_LENGTH:
                    sequence = sequence[
                        :config.MAX_SEQUENCE_LENGTH
                    ]

                valid_pulses.append(
                    (
                        sequence,
                        label,
                        len(sequence),
                        wav_path,
                    )
                )

            if valid_pulses:
                yield valid_pulses


# ============================================================
# BATCH BUILDER
# ============================================================

def build_batches(
    wav_loader,
    batch_size,
    preprocessing_bar
):

    batch = []

    for wav_pulses in wav_loader:

        # One WAV has finished preprocessing.
        preprocessing_bar.update(1)

        for pulse in wav_pulses:

            batch.append(pulse)

            if len(batch) >= batch_size:

                yield collate_batch(batch)

                batch = []

    if batch:
        yield collate_batch(batch)


def collate_batch(batch):

    sequences, labels, lengths, file_paths = zip(*batch)

    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0,
    )

    labels = torch.tensor(
        labels,
        dtype=torch.long,
    )

    lengths = torch.tensor(
        lengths,
        dtype=torch.long,
    )

    return (
        padded_sequences,
        labels,
        lengths,
        file_paths,
    )


# ============================================================
# BACKGROUND BATCH PRODUCER
# ============================================================

def producer(
    wav_loader,
    batch_size,
    output_queue,
    preprocessing_bar,
):

    try:

        for batch in build_batches(
            wav_loader,
            batch_size,
            preprocessing_bar,
        ):
            output_queue.put(batch)

        output_queue.put(None)

    except BaseException as error:

        output_queue.put(error)


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device("cpu")

    print(f"Using device: {device}")
    print(
        f"Preprocessing workers: {NUM_WORKERS}, "
        f"model threads: {MODEL_THREADS}"
    )

    # ========================================================
    # LOAD MODEL CHECKPOINT FIRST
    # ========================================================

    checkpoint_path = (
        Path(__file__).resolve().parent
        / "best_model.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # --------------------------------------------------------
    # Extract model state
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # --------------------------------------------------------
    # Recover the ORIGINAL training class mapping.
    #
    # train.py saves:
    #     "class_names"
    #     "class_to_idx"
    #
    # This is critical because the test directory may contain
    # only a subset of the training classes.
    # --------------------------------------------------------

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "best_model.pt does not contain the training "
            "class mapping. Re-train using the provided "
            "train.py so class_to_idx is saved in the checkpoint."
        )

    if "class_to_idx" not in checkpoint:
        raise RuntimeError(
            "best_model.pt does not contain 'class_to_idx'. "
            "Re-train using train.py so the class mapping is "
            "saved in the checkpoint."
        )

    training_class_to_idx = {
        str(name): int(index)
        for name, index in checkpoint["class_to_idx"].items()
    }

    # Preserve the original training order.
    if "class_names" in checkpoint:

        training_class_names = [
            str(name)
            for name in checkpoint["class_names"]
        ]

    else:

        training_class_names = [
            name
            for name, _ in sorted(
                training_class_to_idx.items(),
                key=lambda x: x[1]
            )
        ]

    num_classes = len(training_class_to_idx)

    # ========================================================
    # LOAD TEST FILES
    # ========================================================

    # IMPORTANT:
    # Pass the TRAINING class_to_idx into discover_wav_files.
    #
    # This means:
    #
    #   training class A -> index 0
    #   training class B -> index 1
    #   ...
    #   training class L -> index 11
    #
    # even when the test directory only contains 8 classes.
    test_samples, _ = discover_wav_files(
        config.TEST_DIR,
        class_to_idx=training_class_to_idx,
    )

    print(
        f"Found {len(test_samples)} test WAV files."
    )

    # --------------------------------------------------------
    # Report which classes are present in the test set
    # --------------------------------------------------------

    test_class_names = sorted(
        {
            Path(file_path).parent.name
            for file_path, _ in test_samples
        }
    )

    print(
        f"Test classes present: "
        f"{len(test_class_names)}/{num_classes}"
    )

    for class_name in training_class_names:

        if class_name in test_class_names:
            print(f"  [TEST] {class_name}")

        else:
            print(f"  [----] {class_name}")

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = LSTMClassifier(
        num_classes=num_classes,
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print(
        f"Loaded model with {num_classes} classes."
    )

    # ========================================================
    # DATASET / DATALOADER
    # ========================================================

    test_dataset = EvaluationWavDataset(
        test_samples,
        config.STATS_FILE,
    )

    loader_kwargs = dict(
        dataset=test_dataset,
        batch_size=None,
        num_workers=NUM_WORKERS,
        worker_init_fn=worker_init,
        pin_memory=False,
        persistent_workers=False,
    )

    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = 2

    test_loader = DataLoader(
        **loader_kwargs
    )

    # ========================================================
    # PROGRESS
    # ========================================================

    preprocessing_bar = tqdm(
        total=len(test_samples),
        desc="Preprocessing",
        unit="wav",
        position=0,
        dynamic_ncols=True,
    )

    evaluation_bar = tqdm(
        total=len(test_samples),
        desc="Evaluating",
        unit="pulse",
        position=1,
        dynamic_ncols=True,
    )

    # ========================================================
    # BACKGROUND QUEUE
    # ========================================================

    batch_queue = queue.Queue(maxsize=4)

    producer_thread = threading.Thread(
        target=producer,
        args=(
            test_loader,
            config.BATCH_SIZE,
            batch_queue,
            preprocessing_bar,
        ),
        daemon=True,
    )

    producer_thread.start()

    # ========================================================
    # RESULTS
    # ========================================================

    all_probabilities = []
    all_logits = []
    all_labels = []
    all_predictions = []
    all_confidences = []
    all_files = []

    # ========================================================
    # INFERENCE
    # ========================================================

    with torch.inference_mode():

        while True:

            item = batch_queue.get()

            if item is None:
                break

            if isinstance(item, BaseException):
                raise item

            (
                sequences,
                labels,
                lengths,
                file_paths,
            ) = item

            sequences = sequences.to(
                device,
                non_blocking=False,
            )

            lengths = lengths.to(
                device,
                non_blocking=False,
            )

            step_logits, final_logits = model(
                sequences,
                lengths,
            )

            # Final attention-pooled logits
            logits = final_logits

            # The model was trained with BCEWithLogitsLoss,
            # so retain the sigmoid probability conversion.
            probabilities = torch.sigmoid(
                logits
            )

            confidences, predictions = (
                probabilities.max(dim=1)
            )

            # ------------------------------------------------
            # Store results
            # ------------------------------------------------

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_logits.append(
                logits.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_confidences.append(
                confidences.cpu().numpy()
            )

            all_files.extend(
                file_paths
            )

            evaluation_bar.update(
                len(file_paths)
            )

    producer_thread.join()

    preprocessing_bar.close()
    evaluation_bar.close()

    # ========================================================
    # RESULTS
    # ========================================================

    if not all_probabilities:
        print("No valid pulses found.")
        return

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    logits = np.concatenate(
        all_logits,
        axis=0,
    )

    labels = np.concatenate(
        all_labels,
        axis=0,
    )

    predictions = np.concatenate(
        all_predictions,
        axis=0,
    )

    confidences = np.concatenate(
        all_confidences,
        axis=0,
    )

    # ========================================================
    # INDEX -> CLASS NAME
    # ========================================================

    idx_to_class = {
        index: name
        for name, index in training_class_to_idx.items()
    }

    ground_truth_names = [
        idx_to_class[int(label)]
        for label in labels
    ]

    prediction_names = [
        idx_to_class[int(prediction)]
        for prediction in predictions
    ]

    # ========================================================
    # OUTPUT DATAFRAME
    # ========================================================

    result = pd.DataFrame({
        "file": [
            Path(file_path).name
            for file_path in all_files
        ],
        "ground_truth": ground_truth_names,
        "prediction": prediction_names,
        "confidence": confidences,
    })

    # ========================================================
    # CLASS PROBABILITY COLUMNS
    #
    # IMPORTANT:
    # Write ALL 12 training classes, not just the 8 classes
    # found in the test folder.
    # ========================================================

    for class_index in range(num_classes):

        class_name = idx_to_class[class_index]

        result[class_name] = probabilities[
            :,
            class_index
        ]

    # ========================================================
    # CLASS LOGIT COLUMNS
    # ========================================================

    for class_index in range(num_classes):

        class_name = idx_to_class[class_index]

        result[
            f"{class_name}_logit"
        ] = logits[
            :,
            class_index
        ]

    # ========================================================
    # SAVE
    # ========================================================

    output_path = (
        Path(__file__).resolve().parent
        / "predictions.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved predictions to:\n"
        f"{output_path}"
    )

    print(
        f"Evaluated {len(result):,} pulses "
        f"from {len(set(all_files)):,} WAV files."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
    