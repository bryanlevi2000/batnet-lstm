
import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

FILTER_CONFIG_PATH = Path(__file__).parent / "pulse_class_filters.json"

# Maps JSON filter names to columns in metrics_df.
METRIC_COLUMNS = {
    "fmin": "fmin_hz",
    "fmax": "fmax_hz",
    "fc": "fc_freq_hz",
    "duration": "duration_us",
}


# ---------------------------------------------------------
# Load class-specific filter configuration
# ---------------------------------------------------------

def load_filter_config(config_path=FILTER_CONFIG_PATH):
    """Load the class-specific pulse metric ranges from JSON."""

    with open(config_path, "r") as file:
        return json.load(file)


# ---------------------------------------------------------
# Determine class from WAV file path
# ---------------------------------------------------------

def get_class_from_path(wav_path):
    """
    The class name is assumed to be the immediate parent
    directory containing the WAV file.
    """

    return Path(wav_path).parent.name


# ---------------------------------------------------------
# Filter pulses for one WAV file
# ---------------------------------------------------------

def filter_file_pulses(pulses_df, metrics_df, class_name, filter_config):
    """
    Filter pulses from one WAV file using the ranges for
    their ground-truth class.

    Returns:
        accepted_pulses: DataFrame containing accepted pulse data.
        accepted_metrics: DataFrame containing accepted metrics.
        stats: Dictionary containing rejection statistics.
    """

    if class_name not in filter_config:
        raise ValueError(
            f"No pulse filter configuration found for class: {class_name}"
        )

    class_filters = filter_config[class_name]

    # Initialize statistics.
    stats = {
        "class": class_name,
        "total_pulses": len(metrics_df),
        "accepted": 0,
        "rejected": 0,
        "rejected_fmin": 0,
        "rejected_fmax": 0,
        "rejected_fc": 0,
        "rejected_duration": 0,
    }

    # Empty input: return correctly structured empty DataFrames.
    if metrics_df.empty:
        return (
            pulses_df.iloc[0:0].copy(),
            metrics_df.iloc[0:0].copy(),
            stats,
        )

    # Start with every pulse accepted.
    accepted_mask = pd.Series(True, index=metrics_df.index)

    # Check each metric against its class-specific range.
    for filter_name, column_name in METRIC_COLUMNS.items():

        if filter_name not in class_filters:
            raise ValueError(
                f"Missing '{filter_name}' filter for class {class_name}"
            )

        if column_name not in metrics_df.columns:
            raise ValueError(
                f"Missing metrics column '{column_name}'"
            )

        lower, upper = class_filters[filter_name]

        values = pd.to_numeric(
            metrics_df[column_name], errors="coerce"
        )

        # NaN values and out-of-range values fail the filter.
        metric_passes = values.between(lower, upper)

        stats[f"rejected_{filter_name}"] = int(
            (~metric_passes).sum()
        )

        accepted_mask &= metric_passes

    # Select accepted metrics.
    accepted_metrics = metrics_df.loc[accepted_mask].copy()

    # Select the corresponding pulse time-series data.
    accepted_ids = accepted_metrics["pulse_id"]

    accepted_pulses = pulses_df[
        pulses_df["pulse_id"].isin(accepted_ids)
    ].copy()

    # Update overall statistics.
    stats["accepted"] = len(accepted_metrics)
    stats["rejected"] = stats["total_pulses"] - stats["accepted"]

    return accepted_pulses, accepted_metrics, stats


# ---------------------------------------------------------
# Filter all WAV files
# ---------------------------------------------------------

def filter_all_files(pulses_by_file, metrics_by_file, config_path=FILTER_CONFIG_PATH):
    """
    Apply class-specific pulse filtering to every WAV file.

    Returns:
        filtered_pulses_by_file
        filtered_metrics_by_file
        rejection_summary
    """

    filter_config = load_filter_config(config_path)

    filtered_pulses_by_file = {}
    filtered_metrics_by_file = {}

    # One row per class in the final rejection summary.
    rejection_summary = {}

    for wav_path, pulses_df in pulses_by_file.items():

        if wav_path not in metrics_by_file:
            raise ValueError(
                f"No metrics DataFrame found for {wav_path}"
            )

        metrics_df = metrics_by_file[wav_path]
        class_name = get_class_from_path(wav_path)

        accepted_pulses, accepted_metrics, stats = filter_file_pulses(
            pulses_df,
            metrics_df,
            class_name,
            filter_config,
        )

        filtered_pulses_by_file[wav_path] = accepted_pulses
        filtered_metrics_by_file[wav_path] = accepted_metrics

        # Accumulate statistics by class.
        if class_name not in rejection_summary:
            rejection_summary[class_name] = {
                "class": class_name,
                "total_pulses": 0,
                "accepted": 0,
                "rejected": 0,
                "rejected_fmin": 0,
                "rejected_fmax": 0,
                "rejected_fc": 0,
                "rejected_duration": 0,
            }

        for key in rejection_summary[class_name]:
            if key != "class":
                rejection_summary[class_name][key] += stats[key]

    return (
        filtered_pulses_by_file,
        filtered_metrics_by_file,
        rejection_summary,
    )


# ---------------------------------------------------------
# Save rejection summary
# ---------------------------------------------------------

def save_rejection_summary(rejection_summary, output_path):
    """Save accumulated class-level rejection statistics to CSV."""

    summary_df = pd.DataFrame(
        list(rejection_summary.values())
    )

    summary_df.to_csv(output_path, index=False)

    return summary_df
