# features.py
import pandas as pd
import numpy as np

def calculate_slope(df):
    """
    Calculates the slope (change in freq_hz / change in time_us) grouped by pulse_id.
    """
    # Calculate differences between consecutive rows for each pulse group
    dfreq = df.groupby('pulse_id')['freq_hz'].diff()
    dtime = df.groupby('pulse_id')['time_us'].diff()
    
    # Raw slope will have NaN for the first point of every group
    raw_slope = dfreq / dtime
    
    # Group by pulse_id and backfill (bfill) to copy the 2nd point's slope to the 1st point
    filled_slope = raw_slope.groupby(df['pulse_id']).bfill()
    
    return filled_slope


def calculate_position(df):
    """
    Calculates the normalized position (0.00 to 1.00) for each pulse, 
    rounded to the nearest 5th percent (0.05).
    """
    # Get the current row index (rank) and the total number of rows (count) per pulse
    rank = df.groupby('pulse_id').cumcount()
    group_size = df.groupby('pulse_id')['pulse_id'].transform('size')
    
    # Calculate raw position from 0.0 to 1.0. (If a pulse only has 1 row, default to 0.0)
    raw_pos = np.where(group_size > 1, rank / (group_size - 1), 0.0)
    
    # Round to the nearest 0.05 (5th percent)
    # Multiplying by 20, rounding, and dividing by 20 snaps values to 0.05 increments
    rounded_pos = np.round(raw_pos * 20) / 20
    
    # Return as a Series aligned with the original DataFrame index
    return pd.Series(rounded_pos, index=df.index, dtype=float)


def process_pulse_dataframe(df):
    """
    Applies the slope and position calculations to the incoming DataFrame,
    returning a new DataFrame with original and new columns.
    """
    result_df = df.copy()
    
    # Apply the two functions
    result_df['slope'] = calculate_slope(result_df)
    result_df['position'] = calculate_position(result_df)
    
    return result_df
