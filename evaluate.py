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
from preprocess import preprocess_wav
from model import LSTMClassifier


# ============================================================
# CPU / WORKER CONFIGURATION
# ============================================================

NUM_WORKERS = config.EVALUATION_NUM_CORES
MODEL_THREADS = config.MODEL_THREADS

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
# RECURSIVE WAV DISCOVERY
# ============================================================

def discover_all_wav_files(root_dir):
    """
    Recursively find every WAV file underneath root_dir.
    """

    root_dir = Path(root_dir)

    if not root_dir.exists():
        raise FileNotFoundError(
            f"TEST_DIR does not exist:\n{root_dir}"
        )

    if not root_dir.is_dir():
        raise NotADirectoryError(
            f"TEST_DIR is not a directory:\n{root_dir}"
        )

    wav_files = sorted(
        path
        for path in root_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".wav"
    )

    return wav_files


# ============================================================
# WAV-LEVEL EVALUATION DATASET
# ============================================================

class EvaluationWavDataset(IterableDataset):

    def __init__(self, wav_files, normalization_path):
        super().__init__()

        self.wav_files = list(wav_files)
        self.normalization_path = normalization_path

    def __iter__(self):

        worker_info = get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        worker_files = self.wav_files[
            worker_id::num_workers
        ]

        for wav_path in worker_files:

            # ------------------------------------------------
            # ALWAYS account for this WAV.
            #
            # If preprocessing fails, do not kill the worker.
            # Instead, yield a failure record.
            # ------------------------------------------------

            try:

                pulses = preprocess_wav(
                    wav_path,
                    normalization_path=self.normalization_path
                )

            except Exception as error:

                yield {
                    "status": "error",
                    "wav_path": wav_path,
                    "error": f"{type(error).__name__}: {error}",
                    "pulses": None,
                }

                continue

            # ------------------------------------------------
            # Process successfully preprocessed WAV
            # ------------------------------------------------

            valid_pulses = []

            try:

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
                            len(sequence),
                            wav_path,
                        )
                    )

            except Exception as error:

                yield {
                    "status": "error",
                    "wav_path": wav_path,
                    "error": (
                        f"{type(error).__name__}: {error}"
                    ),
                    "pulses": None,
                }

                continue

            # ------------------------------------------------
            # WAV successfully processed
            # ------------------------------------------------

            yield {
                "status": "success",
                "wav_path": wav_path,
                "error": None,
                "pulses": valid_pulses,
            }


# ============================================================
# BATCH BUILDER
# ============================================================

def build_batches(
    wav_loader,
    batch_size,
    preprocessing_bar,
    error_records,
    successful_wavs,
    zero_pulse_wavs,
):

    batch = []

    for wav_result in wav_loader:

        # ----------------------------------------------------
        # Every result corresponds to exactly one WAV.
        # Therefore this progress bar represents WAVs actually
        # accounted for, rather than WAVs that happened to
        # produce pulses.
        # ----------------------------------------------------

        preprocessing_bar.update(1)

        status = wav_result["status"]
        wav_path = wav_result["wav_path"]

        # ----------------------------------------------------
        # Failed WAV
        # ----------------------------------------------------

        if status == "error":

            error_records.append({
                "file": str(wav_path),
                "status": "preprocessing_error",
                "error": wav_result["error"],
            })

            continue

        # ----------------------------------------------------
        # Successful WAV
        # ----------------------------------------------------

        successful_wavs.append(wav_path)

        wav_pulses = wav_result["pulses"]

        # A WAV can preprocess successfully but contain
        # no usable pulses.
        if not wav_pulses:
            zero_pulse_wavs.append(wav_path)

            error_records.append({
                "file": str(wav_path),
                "status": "no_pulses",
                "error": "Preprocessing succeeded, but no usable pulses were produced.",
            })

            continue
        # ----------------------------------------------------
        # Add pulses to batches
        # ----------------------------------------------------

        for pulse in wav_pulses:

            batch.append(pulse)

            if len(batch) >= batch_size:

                yield collate_batch(batch)

                batch = []

    # --------------------------------------------------------
    # Final partial batch
    # --------------------------------------------------------

    if batch:
        yield collate_batch(batch)


# ============================================================
# COLLATE
# ============================================================

def collate_batch(batch):

    sequences, lengths, file_paths = zip(*batch)

    padded_sequences = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0,
    )

    lengths = torch.tensor(
        lengths,
        dtype=torch.long,
    )

    return (
        padded_sequences,
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
    error_records,
    successful_wavs,
    zero_pulse_wavs,
):

    try:

        for batch in build_batches(
            wav_loader,
            batch_size,
            preprocessing_bar,
            error_records,
            successful_wavs,
            zero_pulse_wavs,
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
    # LOAD MODEL CHECKPOINT
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

    # ========================================================
    # RECOVER MODEL CLASS MAPPING
    # ========================================================

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "best_model.pt does not contain the training "
            "class mapping. Re-train using train.py so "
            "class_to_idx is saved in the checkpoint."
        )

    if "class_to_idx" not in checkpoint:
        raise RuntimeError(
            "best_model.pt does not contain 'class_to_idx'. "
            "Re-train using train.py so the class mapping "
            "is saved in the checkpoint."
        )

    training_class_to_idx = {
        str(name): int(index)
        for name, index in checkpoint["class_to_idx"].items()
    }

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
    # LOAD ALL WAV FILES RECURSIVELY
    # ========================================================

    wav_files = discover_all_wav_files(
        config.TEST_DIR
    )

    print(
        f"Found {len(wav_files):,} WAV files."
    )

    if not wav_files:
        print(
            f"No WAV files found under:\n"
            f"{Path(config.TEST_DIR).resolve()}"
        )
        return

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
        wav_files,
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
        total=len(wav_files),
        desc="Preprocessing",
        unit="wav",
        position=0,
        dynamic_ncols=True,
    )

    evaluation_bar = tqdm(
        desc="Evaluating",
        unit="pulse",
        position=1,
        dynamic_ncols=True,
    )

    # ========================================================
    # BACKGROUND QUEUE
    # ========================================================

    batch_queue = queue.Queue(maxsize=4)

    # These are written by the producer thread.
    error_records = []
    successful_wavs = []
    zero_pulse_wavs = []

    producer_thread = threading.Thread(
        target=producer,
        args=(
            test_loader,
            config.BATCH_SIZE,
            batch_queue,
            preprocessing_bar,
            error_records,
            successful_wavs,
            zero_pulse_wavs,
        ),
        daemon=True,
    )

    producer_thread.start()

    # ========================================================
    # RESULTS
    # ========================================================

    all_probabilities = []
    all_logits = []
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
    # SAVE PREPROCESSING ERRORS
    # ========================================================

    output_directory = (
        Path(__file__).resolve().parent
    )

    errors_path = (
        output_directory
        / "evaluation_errors.csv"
    )

    if error_records:

        error_result = pd.DataFrame(
            error_records
        )

        error_result.to_csv(
            errors_path,
            index=False,
        )

    else:

        # Still create the file so the user can confirm
        # that there were zero preprocessing errors.
        pd.DataFrame(
            columns=[
                "file",
                "status",
                "error",
            ]
        ).to_csv(
            errors_path,
            index=False,
        )

    # ========================================================
    # RESULTS
    # ========================================================

    if not all_probabilities:

        print(
            "\nNo valid pulses found."
        )

        print(
            f"Found:              {len(wav_files):,} WAV files"
        )

        print(
            f"Successfully read:  {len(successful_wavs):,} WAV files"
        )

        print(
            f"Preprocessing errors:{len(error_records):,} WAV files"
        )

        print(
            f"No usable pulses:   {len(zero_pulse_wavs):,} WAV files"
        )

        print(
            f"\nSaved error report to:\n"
            f"{errors_path}"
        )

        return

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    logits = np.concatenate(
        all_logits,
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

    prediction_names = [
        idx_to_class[int(prediction)]
        for prediction in predictions
    ]

    # ========================================================
    # GROUND TRUTH FROM IMMEDIATE PARENT DIRECTORY
    # ========================================================

    ground_truth_names = [
        Path(file_path).parent.name
        for file_path in all_files
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
    # SAVE PREDICTIONS
    # ========================================================

    output_path = (
        output_directory
        / "predictions.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    evaluated_wavs = len(
        set(all_files)
    )

    print(
        f"\nSaved predictions to:\n"
        f"{output_path}"
    )

    print(
        f"\nSaved preprocessing error report to:\n"
        f"{errors_path}"
    )

    print("\nEvaluation summary:")
    print(
        f"  WAV files found:          {len(wav_files):,}"
    )
    print(
        f"  WAV files successfully read: "
        f"{len(successful_wavs):,}"
    )
    print(
        f"  WAV files with errors:    "
        f"{len(error_records):,}"
    )
    print(
        f"  WAV files with no pulses: "
        f"{len(zero_pulse_wavs):,}"
    )
    print(
        f"  WAV files producing predictions: "
        f"{evaluated_wavs:,}"
    )
    print(
        f"  Pulses evaluated:         "
        f"{len(result):,}"
    )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    accounted_for = (
        len(successful_wavs)
        + len(error_records)
    )

    if accounted_for != len(wav_files):

        print(
            "\nWARNING:"
        )

        print(
            f"Only {accounted_for:,} of "
            f"{len(wav_files):,} WAV files were accounted for."
        )

    else:

        print(
            "\nAll discovered WAV files were accounted for."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()