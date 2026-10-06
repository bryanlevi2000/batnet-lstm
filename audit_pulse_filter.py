from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import argparse
import csv

import config

from preprocess import (
    process_wav,
    DEFAULT_CLASS_FILTER_CONFIG,
)


# ============================================================
# PROCESS ONE WAV
# ============================================================

def audit_one_wav(
    wav_path,
    class_filter_config_path
):

    wav_path = Path(wav_path)

    try:

        _, stats = process_wav(
            wav_path,
            apply_class_filter=True,
            class_filter_config_path=(
                class_filter_config_path
            ),
            return_filter_stats=True,
            stop_after_class_filter=True
        )

        stats["wav_path"] = str(wav_path)
        stats["error"] = ""

        return stats

    except Exception as exc:

        return {
            "wav_path": str(wav_path),
            "class_name": wav_path.parent.name,
            "candidate_pulses": 0,
            "accepted_pulses": 0,
            "rejected_pulses": 0,
            "rejected_fmin": 0,
            "rejected_fmax": 0,
            "rejected_fc_range": 0,
            "rejected_missing_fc": 0,
            "error": repr(exc),
        }


# ============================================================
# FIND WAV FILES
# ============================================================

def find_wav_files(root_dir):

    root_dir = Path(root_dir)

    return sorted(
        [
            path
            for path in root_dir.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() == ".wav"
            )
        ]
    )


# ============================================================
# AGGREGATE BY CLASS
# ============================================================

def aggregate_by_class(results):

    totals = defaultdict(
        lambda: {
            "wav_files": 0,
            "candidate_pulses": 0,
            "accepted_pulses": 0,
            "rejected_pulses": 0,
            "rejected_fmin": 0,
            "rejected_fmax": 0,
            "rejected_fc_range": 0,
            "rejected_missing_fc": 0,
            "errors": 0,
        }
    )

    for result in results:

        class_name = result["class_name"]

        summary = totals[class_name]

        if result["error"]:
            summary["errors"] += 1
            continue

        summary["wav_files"] += 1

        summary["candidate_pulses"] += (
            result["candidate_pulses"]
        )

        summary["accepted_pulses"] += (
            result["accepted_pulses"]
        )

        summary["rejected_pulses"] += (
            result["rejected_pulses"]
        )

        summary["rejected_fmin"] += (
            result["rejected_fmin"]
        )

        summary["rejected_fmax"] += (
            result["rejected_fmax"]
        )

        summary["rejected_fc_range"] += (
            result["rejected_fc_range"]
        )

        summary["rejected_missing_fc"] += (
            result["rejected_missing_fc"]
        )

    return totals


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(totals):

    print()
    print("=" * 125)
    print("CLASS-SPECIFIC PULSE FILTER AUDIT")
    print("=" * 125)

    header = (
        f"{'Class':<25}"
        f"{'WAVs':>8}"
        f"{'Candidates':>12}"
        f"{'Accepted':>12}"
        f"{'Rejected':>12}"
        f"{'Fmin fail':>12}"
        f"{'Fmax fail':>12}"
        f"{'Fc fail':>12}"
        f"{'No Fc':>10}"
        f"{'Accept %':>10}"
    )

    print(header)
    print("-" * 125)

    total_candidates = 0
    total_accepted = 0
    total_rejected = 0

    for class_name in sorted(totals):

        summary = totals[class_name]

        candidates = summary["candidate_pulses"]
        accepted = summary["accepted_pulses"]
        rejected = summary["rejected_pulses"]

        if candidates > 0:
            acceptance_rate = (
                100.0 * accepted / candidates
            )
        else:
            acceptance_rate = 0.0

        print(
            f"{class_name:<25}"
            f"{summary['wav_files']:>8,}"
            f"{candidates:>12,}"
            f"{accepted:>12,}"
            f"{rejected:>12,}"
            f"{summary['rejected_fmin']:>12,}"
            f"{summary['rejected_fmax']:>12,}"
            f"{summary['rejected_fc_range']:>12,}"
            f"{summary['rejected_missing_fc']:>10,}"
            f"{acceptance_rate:>9.2f}%"
        )

        total_candidates += candidates
        total_accepted += accepted
        total_rejected += rejected

    print("-" * 125)

    if total_candidates > 0:
        total_acceptance_rate = (
            100.0
            * total_accepted
            / total_candidates
        )
    else:
        total_acceptance_rate = 0.0

    print(
        f"{'TOTAL':<25}"
        f"{sum(v['wav_files'] for v in totals.values()):>8,}"
        f"{total_candidates:>12,}"
        f"{total_accepted:>12,}"
        f"{total_rejected:>12,}"
        f"{sum(v['rejected_fmin'] for v in totals.values()):>12,}"
        f"{sum(v['rejected_fmax'] for v in totals.values()):>12,}"
        f"{sum(v['rejected_fc_range'] for v in totals.values()):>12,}"
        f"{sum(v['rejected_missing_fc'] for v in totals.values()):>10,}"
        f"{total_acceptance_rate:>9.2f}%"
    )

    print("=" * 125)

    print()
    print(
        "NOTE: Fmin/Fmax/Fc failure counts are not mutually "
        "exclusive. A single pulse can fail multiple criteria."
    )


# ============================================================
# WRITE CLASS SUMMARY CSV
# ============================================================

def write_class_summary(
    totals,
    output_path
):

    fieldnames = [
        "class_name",
        "wav_files",
        "candidate_pulses",
        "accepted_pulses",
        "rejected_pulses",
        "rejected_fmin",
        "rejected_fmax",
        "rejected_fc_range",
        "rejected_missing_fc",
        "acceptance_percent",
        "errors",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for class_name in sorted(totals):

            summary = totals[class_name]

            candidates = summary["candidate_pulses"]

            if candidates > 0:

                acceptance_percent = (
                    100.0
                    * summary["accepted_pulses"]
                    / candidates
                )

            else:

                acceptance_percent = 0.0

            writer.writerow({
                "class_name": class_name,
                "wav_files": summary["wav_files"],
                "candidate_pulses": candidates,
                "accepted_pulses": summary["accepted_pulses"],
                "rejected_pulses": summary["rejected_pulses"],
                "rejected_fmin": summary["rejected_fmin"],
                "rejected_fmax": summary["rejected_fmax"],
                "rejected_fc_range": summary["rejected_fc_range"],
                "rejected_missing_fc": summary["rejected_missing_fc"],
                "acceptance_percent": (
                    f"{acceptance_percent:.4f}"
                ),
                "errors": summary["errors"],
            })


# ============================================================
# WRITE FILE-LEVEL RESULTS
# ============================================================

def write_file_results(
    results,
    output_path
):

    fieldnames = [
        "wav_path",
        "class_name",
        "candidate_pulses",
        "accepted_pulses",
        "rejected_pulses",
        "rejected_fmin",
        "rejected_fmax",
        "rejected_fc_range",
        "rejected_missing_fc",
        "error",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Audit the class-specific pulse filter."
        )
    )

    parser.add_argument(
        "--root",
        default=str(config.TRAIN_DIR),
        help=(
            "Root directory containing class subdirectories. "
            "Defaults to config.TRAIN_DIR."
        )
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=config.NUM_WORKERS,
        help=(
            "Number of worker processes. "
            "Defaults to config.NUM_WORKERS."
        )
    )

    args = parser.parse_args()

    root_dir = Path(args.root).resolve()

    class_filter_config_path = (
        DEFAULT_CLASS_FILTER_CONFIG
    )

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not root_dir.exists():

        raise FileNotFoundError(
            f"Root directory does not exist:\n"
            f"{root_dir}"
        )

    if not class_filter_config_path.exists():

        raise FileNotFoundError(
            f"Class filter JSON does not exist:\n"
            f"{class_filter_config_path}"
        )

    # --------------------------------------------------------
    # Find WAVs
    # --------------------------------------------------------

    wav_files = find_wav_files(
        root_dir
    )

    print(
        f"Found {len(wav_files):,} WAV files under:\n"
        f"{root_dir}"
    )

    print(
        f"Using class filter:\n"
        f"{class_filter_config_path}"
    )

    if not wav_files:
        print("No WAV files found.")
        return

    # --------------------------------------------------------
    # Process WAVs
    # --------------------------------------------------------

    worker_count = max(
        1,
        args.workers
    )

    worker_args = [
        (
            wav_path,
            class_filter_config_path
        )
        for wav_path in wav_files
    ]

    if worker_count == 1:

        results = [
            audit_one_wav(
                wav_path,
                class_filter_config_path
            )
            for wav_path in wav_files
        ]

    else:

        with ProcessPoolExecutor(
            max_workers=worker_count
        ) as executor:

            results = list(
                executor.map(
                    audit_worker,
                    worker_args
                )
            )

    # --------------------------------------------------------
    # Check errors
    # --------------------------------------------------------

    errors = [
        result
        for result in results
        if result["error"]
    ]

    if errors:

        print()
        print(
            f"WARNING: {len(errors):,} WAV files "
            f"failed during auditing."
        )

        for result in errors[:20]:

            print(
                f"\n{result['wav_path']}"
                f"\n  {result['error']}"
            )

        if len(errors) > 20:
            print(
                f"\n...and {len(errors) - 20:,} more errors."
            )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    totals = aggregate_by_class(
        results
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_summary(
        totals
    )

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    output_dir = (
        Path(__file__).resolve().parent
    )

    class_summary_path = (
        output_dir
        / "pulse_filter_audit_by_class.csv"
    )

    file_results_path = (
        output_dir
        / "pulse_filter_audit_by_file.csv"
    )

    write_class_summary(
        totals,
        class_summary_path
    )

    write_file_results(
        results,
        file_results_path
    )

    print()
    print(
        f"Saved class summary to:\n"
        f"{class_summary_path}"
    )

    print(
        f"Saved file-level results to:\n"
        f"{file_results_path}"
    )

def audit_worker(args):

    wav_path, class_filter_config_path = args

    return audit_one_wav(
        wav_path,
        class_filter_config_path
    )

if __name__ == "__main__":
    main()