# normalize.py
import json
from pathlib import Path

import numpy as np


def update_normalizer(stats, df, columns):
    """
    Update running normalization statistics using one processed
    pulse dataframe.

    Only rows with a valid pulse_id are included.

    Parameters
    ----------
    stats : dict
        Running statistics returned by a previous call.
    df : pandas.DataFrame
        Processed pulse dataframe.
    columns : list[str]
        Columns to normalize.

    Returns
    -------
    dict
        Updated running statistics.
    """

    pulse_df = df[df["pulse_id"].notna()]

    for column in columns:

        values = pulse_df[column].dropna().to_numpy(dtype=float)

        if len(values) == 0:
            continue

        # Initialize column statistics
        if column not in stats:
            stats[column] = {
                "count": 0,
                "mean": 0.0,
                "M2": 0.0,
            }

        column_stats = stats[column]

        for value in values:

            column_stats["count"] += 1

            delta = value - column_stats["mean"]

            column_stats["mean"] += (
                delta / column_stats["count"]
            )

            delta2 = value - column_stats["mean"]

            column_stats["M2"] += delta * delta2

    return stats


def finalize_normalizer(stats, output_path):
    """
    Convert running statistics into mean/std normalization parameters
    and save them to JSON.
    """

    normalization_params = {}

    for column, values in stats.items():

        count = values["count"]

        if count < 2:
            raise ValueError(
                f"Not enough valid values to calculate "
                f"standard deviation for '{column}'."
            )

        mean = values["mean"]

        # Sample standard deviation
        std = np.sqrt(
            values["M2"] / (count - 1)
        )

        if std == 0:
            raise ValueError(
                f"Standard deviation for '{column}' is zero."
            )

        normalization_params[column] = {
            "mean": float(mean),
            "std": float(std),
            "count": count,
        }

    output_path = Path(output_path)

    with output_path.open("w") as f:
        json.dump(
            normalization_params,
            f,
            indent=4
        )

    return normalization_params


def apply_normalizer(df, normalization_path):
    """
    Apply previously calculated normalization parameters to a dataframe.

    Only rows with a valid pulse_id are normalized.

    New columns are created using the original column name plus
    '_normalized'.
    """

    with open(normalization_path, "r") as f:
        normalization_params = json.load(f)

    result = df.copy()

    pulse_mask = result["pulse_id"].notna()

    for column, params in normalization_params.items():

        normalized_column = f"{column}_normalized"

        result.loc[pulse_mask, normalized_column] = (
            result.loc[pulse_mask, column]
            - params["mean"]
        ) / params["std"]

    return result
