# normalize.py
import json
from pathlib import Path

import numpy as np

import config


NORMALIZATION_STATS_PATH = Path(__file__).with_name(
    "normalization_stats.json"
)


def normalize_dataframe(df):
    """
    Normalize the configured columns of a DataFrame using z-score
    normalization.

    Args:
        df (pd.DataFrame): DataFrame containing the features to normalize.

    Returns:
        pd.DataFrame: A copy of the DataFrame with normalized columns.
    """

    # Load normalization statistics.
    with open(
        NORMALIZATION_STATS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        normalization_stats = json.load(file)

    # Work on a copy to avoid modifying the original DataFrame.
    normalized_df = df.copy()

    # Normalize configured columns.
    for column in config.NORMALIZATION_COLUMNS:

        if column not in normalized_df.columns:
            raise ValueError(
                f"Column '{column}' was not found in the DataFrame."
            )

        if column not in normalization_stats:
            raise ValueError(
                f"Normalization statistics for '{column}' "
                f"were not found in the JSON file."
            )

        mean = normalization_stats[column]["mean"]
        std = normalization_stats[column]["std"]

        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError(
                f"Invalid normalization statistics for '{column}': "
                f"mean={mean}, std={std}."
            )

        normalized_df[column] = (
            normalized_df[column] - mean
        ) / std

    return normalized_df
