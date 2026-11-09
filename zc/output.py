# output.py

from pathlib import Path
import pandas as pd


def write_pulses(
    df: pd.DataFrame,
    output_dir,
    source_path,
    input_dir,
    output_mode="individual"
):
    """
    Write accepted pulses either as individual CSV files or as one
    combined CSV file per WAV file.

    Parameters
    ----------
    df : pandas.DataFrame
        Point-level dataframe containing a 'pulse_id' column.

    output_dir : str or Path
        Root directory where CSVs will be written.

    source_path : str or Path
        Path of the original WAV file being processed.

    input_dir : str or Path
        Root input directory. The relative directory structure of the
        original WAV file will be preserved in the output directory.

    output_mode : str
        "individual" = one CSV per pulse.
        "combined" = one CSV per WAV file.

    Returns
    -------
    int
        Number of pulses represented in the output.
    """

    df = df.copy()

    output_dir = Path(output_dir)
    source_path = Path(source_path)
    input_dir = Path(input_dir)

    if output_mode not in ("individual", "combined"):
        raise ValueError(
            f"Invalid output_mode: {output_mode!r}. "
            "Must be 'individual' or 'combined'."
        )

    # Determine the original file's location relative to the input directory
    relative_dir = source_path.parent.relative_to(input_dir)

    # Create corresponding output directory
    target_dir = output_dir / relative_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    # Original WAV filename without extension
    base_name = source_path.stem

    # ------------------------------------------------------------
    # Combined mode
    # ------------------------------------------------------------

    if output_mode == "combined":

        output_filename = f"{base_name}.csv"
        output_path = target_dir / output_filename

        # If there are no accepted pulses, create an empty dataframe
        # with the expected columns.
        if df.empty:
            df = pd.DataFrame(
                columns=[
                    "time_us",
                    "freq_hz",
                    "amplitude",
                    "pulse_id",
                    "status"
                ]
            )

        df.to_csv(output_path, index=False)

        # Number of pulses represented in the file
        if df.empty:
            return 0

        return df["pulse_id"].nunique()

    # ------------------------------------------------------------
    # Individual mode
    # ------------------------------------------------------------

    if df.empty:
        return 0

    if "pulse_id" not in df.columns:
        raise ValueError("Dataframe must contain a 'pulse_id' column.")

    pulse_count = 0

    for pulse_id, pulse in df.groupby("pulse_id", sort=True):

        pulse = pulse.copy()

        output_filename = f"{base_name}_pulse_{pulse_id}.csv"
        output_path = target_dir / output_filename

        pulse.to_csv(output_path, index=False)

        pulse_count += 1

    return pulse_count
