# _optional_get_metrics.py

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import time

import librosa
import pandas as pd
from tqdm import tqdm

# Add the parent directory (batnet-lstm) to sys.path so Python can find config.py
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

# Local module imports (in the 'zc' folder)
from denoise import denoise_audio
from zc_extraction import extract_zc
from pulse_detection import filter_zc_linearity, greedy_walk
from pulse_cleanup import filter_pulse_tails, clean_pulses

# Pulse metrics module imports
from pulse_metrics import (
    filter_valid_status,
    calculate_slopes,
    calculate_fc,
    calculate_knee,
    calculate_duration,
    calculate_fmax,
    calculate_fmin,
    calculate_tstart,
    calculate_tend,
)


# ============================================================
# PROCESS ONE WAV FILE TO METRICS DATAFRAME
# ============================================================

def process_wav_to_metrics(source_path):
    source_path = Path(source_path)

    times = {
        "read_wav": 0.0,
        "denoising": 0.0,
        "zc_extraction": 0.0,
        "local_linearity": 0.0,
        "greedy_walk": 0.0,
        "pulse_tail_filter": 0.0,
        "pulse_filtering": 0.0,
        "metrics_calculation": 0.0,
    }

    # --------------------------------------------------------
    # 1. Read WAV
    # --------------------------------------------------------
    start = time.perf_counter()
    y, sample_rate = librosa.load(source_path, sr=None, mono=True)
    times["read_wav"] = time.perf_counter() - start

    # --------------------------------------------------------
    # 2. Denoising
    # --------------------------------------------------------
    start = time.perf_counter()
    y = denoise_audio(y, noise_factor=config.NOISE_FACTOR, noise_stride=config.NOISE_STRIDE)
    y = (y * 32767).astype("int16")
    times["denoising"] = time.perf_counter() - start

    # --------------------------------------------------------
    # 3. Zero-crossing extraction
    # --------------------------------------------------------
    start = time.perf_counter()
    zc_data = extract_zc(
        y,
        sample_rate,
        divrat=config.DIVRAT,
        hysteresis_percentile=config.HYSTERESIS_PERCENTILE
    )
    df = pd.DataFrame(zc_data, columns=["time_us", "freq_hz", "amplitude"])
    times["zc_extraction"] = time.perf_counter() - start

    # --------------------------------------------------------
    # 4. Local linearity
    # --------------------------------------------------------
    start = time.perf_counter()
    df = filter_zc_linearity(
        df,
        lookahead_n=config.LOOKAHEAD_N,
        linearity_cutoff=config.LINEARITY_CUTOFF
    )
    times["local_linearity"] = time.perf_counter() - start

    # --------------------------------------------------------
    # 5. Greedy walk
    # --------------------------------------------------------
    start = time.perf_counter()
    df = greedy_walk(
        df,
        time_cutoff=config.TIME_CUTOFF,
        inc_freq_cutoff=config.INC_FREQ_CUTOFF_HZ,
        dec_freq_cutoff=config.DEC_FREQ_CUTOFF_HZ,
        min_points=config.MIN_POINTS,
        buffer_us=config.BUFFER_US
    )
    times["greedy_walk"] = time.perf_counter() - start

    # --------------------------------------------------------
    # 6. Frequency-decrease cutoff in pulse tails
    # --------------------------------------------------------
    start = time.perf_counter()
    df = filter_pulse_tails(
        df,
        tail_time_fraction=config.TAIL_TIME_FRACTION,
        tail_dec_freq_cutoff=config.TAIL_DEC_FREQ_CUTOFF_HZ
    )
    times["pulse_tail_filter"] = time.perf_counter() - start

    # --------------------------------------------------------
    # 7. Pulse metric calculation and filtering
    # --------------------------------------------------------
    start = time.perf_counter()
    df = clean_pulses(
        df,
        min_bandwidth_hz=config.MIN_BANDWIDTH_HZ,
        max_bandwidth_hz=config.MAX_BANDWIDTH_HZ,
        min_duration_us=config.MIN_DURATION_US,
        max_duration_us=config.MAX_DURATION_US,
        fmin_ranges=config.FMIN_RANGES,
        fmax_ranges=config.FMAX_RANGES
    )
    times["pulse_filtering"] = time.perf_counter() - start

    # --------------------------------------------------------
    # 8. Compute Pulse Metrics via pulse_metrics.py
    # --------------------------------------------------------
    start = time.perf_counter()

    # Filter valid status pulses
    df_valid = filter_valid_status(df)

    if df_valid.empty:
        times["metrics_calculation"] = time.perf_counter() - start
        return times, None

    # Ensure slopes are calculated if needed by downstream functions
    df_valid = calculate_slopes(df_valid)

    # Fetch configuration proportions or use sensible defaults
    fc_prop = getattr(config, "METRIC_FC_PROPORTION", 0.2)
    knee_prop = getattr(config, "METRIC_KNEE_PROPORTION", 0.2)

    # Compute individual metric DataFrames
    df_duration = calculate_duration(df_valid).rename(columns={"duration_us": "duration"})
    df_fmax = calculate_fmax(df_valid).rename(columns={"fmax_hz": "fmax"})
    df_fmin = calculate_fmin(df_valid).rename(columns={"fmin_hz": "fmin"})
    df_tstart = calculate_tstart(df_valid).rename(columns={"tstart_us": "tstart"})
    df_tend = calculate_tend(df_valid).rename(columns={"tend_us": "tend"})
    
    # Keep original column name "fc_idx" so calculate_knee can read it
    df_fc = calculate_fc(df_valid, fc_prop)
    df_knee = calculate_knee(df_valid, df_fc, knee_prop)

    # Now rename them to match your desired master schema ("fc" and "knee")
    df_fc = df_fc.rename(columns={"fc_idx": "fc"})
    df_knee = df_knee.rename(columns={"knee_idx": "knee"})

    # Merge all metrics together on pulse_id
    metrics_df = df_duration.merge(df_fmax, on="pulse_id", how="left")
    metrics_df = metrics_df.merge(df_fmin, on="pulse_id", how="left")
    metrics_df = metrics_df.merge(df_fc, on="pulse_id", how="left")
    metrics_df = metrics_df.merge(df_knee, on="pulse_id", how="left")
    metrics_df = metrics_df.merge(df_tstart, on="pulse_id", how="left")
    metrics_df = metrics_df.merge(df_tend, on="pulse_id", how="left")

    # Insert file path column at the front
    metrics_df.insert(0, "file", str(source_path))

    times["metrics_calculation"] = time.perf_counter() - start

    return times, metrics_df


# ============================================================
# MAIN
# ============================================================

def main():
    input_dir = Path(config.INPUT_DIR)

    wav_files = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    ]

    print(f"Found {len(wav_files):,} WAV files.")

    if not wav_files:
        return

    total_times = {
        "read_wav": 0.0,
        "denoising": 0.0,
        "zc_extraction": 0.0,
        "local_linearity": 0.0,
        "greedy_walk": 0.0,
        "pulse_tail_filter": 0.0,
        "pulse_filtering": 0.0,
        "metrics_calculation": 0.0,
    }

    all_file_dfs = []
    total_pulses = 0

    wall_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=config.NUM_WORKERS) as executor:
        futures = [
            executor.submit(process_wav_to_metrics, wav_path)
            for wav_path in wav_files
        ]

        with tqdm(total=len(futures), desc="Processing WAV files", unit="file") as progress:
            for future in as_completed(futures):
                times, file_metrics_df = future.result()

                for step in total_times:
                    total_times[step] += times[step]

                if file_metrics_df is not None and not file_metrics_df.empty:
                    all_file_dfs.append(file_metrics_df)
                    total_pulses += len(file_metrics_df)

                progress.update(1)

    wall_time = time.perf_counter() - wall_start

    # --------------------------------------------------------
    # Combine and Save Master DataFrame
    # --------------------------------------------------------
    if all_file_dfs:
        master_df = pd.concat(all_file_dfs, ignore_index=True)
        
        # Ensure output directory exists and save master CSV
        output_csv_path = Path(getattr(config, "METRIC_OUTPUT_FILE", "pulse_metrics.csv"))
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        master_df.to_csv(output_csv_path, index=False)
        print(f"\nSaved master metrics DataFrame to: {output_csv_path.resolve()}")
    else:
        print("\nNo valid pulses found across all files.")

    # --------------------------------------------------------
    # Timing Report
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE TIMING REPORT")
    print("=" * 60)

    print(f"\nFiles processed: {len(wav_files):,}")
    print(f"Total pulses:    {total_pulses:,}")

    print("\nCumulative processing time:")
    print("-" * 60)

    labels = {
        "read_wav": "Reading WAV files",
        "denoising": "Denoising",
        "zc_extraction": "Zero-crossing extraction",
        "local_linearity": "Local linearity calculation",
        "greedy_walk": "Greedy walk",
        "pulse_tail_filter": "Pulse tail filtering",
        "pulse_filtering": "Pulse filtering/status labeling",
        "metrics_calculation": "Pulse metrics calculation",
    }

    for step, label in labels.items():
        seconds = total_times[step]
        print(f"{label:<35} {seconds:>12.3f} seconds")

    print("-" * 60)
    print(f"{'TOTAL CUMULATIVE STAGE TIME':<35} {sum(total_times.values()):>12.3f} seconds")
    print(f"{'TOTAL WALL-CLOCK TIME':<35} {wall_time:>12.3f} seconds")
    print("=" * 60)


if __name__ == "__main__":
    main()
