# features.py
import pandas as pd
import numpy as np

def calculate_slope(df):
    """
    Calculates the slope (change in freq_hz / change in time_us) for valid rows.
    Valid rows: pulse_id is not NaN and status > 1.
    """
    # Initialize a series with NaN
    slope_series = pd.Series(np.nan, index=df.index, dtype=float)
    
    # Filter for the target conditions
    mask = df['pulse_id'].notna() & (df['status'] > 1)
    valid_df = df[mask]
    
    # Calculate differences between consecutive rows for each pulse group
    dfreq = valid_df.groupby('pulse_id')['freq_hz'].diff()
    dtime = valid_df.groupby('pulse_id')['time_us'].diff()
    
    # Raw slope will have NaN for the first point of every group
    raw_slope = dfreq / dtime
    
    # Assign the raw slope, group by pulse, and backfill (bfill) 
    # to copy the 2nd point's slope to the 1st point
    filled_slope = raw_slope.groupby(valid_df['pulse_id']).bfill()
    
    # Map the calculated slopes back to the original index
    slope_series.loc[mask] = filled_slope
    return slope_series


def calculate_position(df):
    """
    Calculates the normalized position (0.00 to 1.00) for each pulse, 
    rounded to the nearest 5th percent (0.05).
    """
    # Initialize a series with NaN
    position_series = pd.Series(np.nan, index=df.index, dtype=float)
    
    # Filter for the target conditions
    mask = df['pulse_id'].notna() & (df['status'] > 1)
    valid_df = df[mask]
    
    # Get the current row index (rank) and the total number of rows (count) per pulse
    rank = valid_df.groupby('pulse_id').cumcount()
    group_size = valid_df.groupby('pulse_id')['pulse_id'].transform('size')
    
    # Calculate raw position from 0.0 to 1.0. (If a pulse only has 1 row, default to 0.0)
    raw_pos = np.where(group_size > 1, rank / (group_size - 1), 0.0)
    
    # Round to the nearest 0.05 (5th percent)
    # Multiplying by 20, rounding, and dividing by 20 snaps values to 0.05 increments
    rounded_pos = np.round(raw_pos * 20) / 20
    
    # Map the positions back to the original index
    position_series.loc[mask] = rounded_pos
    return position_series


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
