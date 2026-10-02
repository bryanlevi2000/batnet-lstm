# config.py

# Data Preprocessing 
# ============================================================
# INPUT / OUTPUT
# ============================================================
TRAIN_DIR = r"C:\Users\Students\Downloads\081126V1\1_Train"
TEST_DIR = r"C:\Users\Students\Box\BatNET 2024-2026\Bat Recordings (Call Library)\Subsets for Development\Test Sets\usfws_12_V2"

# ============================================================
# DENOISING
# ============================================================
NOISE_FACTOR = 2.0
NOISE_STRIDE = 4

# ============================================================
# ZC EXTRACTION
# ============================================================
DIVRAT = 8
HYSTERESIS_PERCENTILE = 85.0

# ============================================================
# PULSE DETECTION
# ============================================================
# Local linearity
LOOKAHEAD_N = 15
LINEARITY_CUTOFF = 0.9

# Greedy walk
TIME_CUTOFF = 500  # microseconds
INC_FREQ_CUTOFF_HZ = 1000
DEC_FREQ_CUTOFF_HZ = 5000
MIN_POINTS = 15
BUFFER_US = 10000

# ============================================================
# PULSE FILTERING
# ============================================================
# Characteristic frequency
CHARACTERISTIC_MIN_POINTS = 2
CHARACTERISTIC_LAST_FRACTION = 0.33

# Maximum allowed frequency decrease after characteristic frequency
CHARACTERISTIC_FREQUENCY_DECREASE_CUTOFF_HZ = 1000

# Pulse metrics
MIN_BANDWIDTH_HZ = 2000
MAX_BANDWIDTH_HZ = 120000

MIN_DURATION_US = 100
MAX_DURATION_US = 30000

FMIN_RANGES = (5000, 50000)
FMAX_RANGES = (10000, 130000)

# ============================================================
# PCHP INTERPOLATION
# ============================================================
PCHIP_TIME_INTERVAL = 100  # microseconds

# ============================================================
# NORMALIZATION
# ============================================================
COLUMNS = ["freq_hz", "amplitude"] # options are slope, freq_hz, amplitude, position
STATS_FILE = r"C:\Users\Students\Box\BatNET 2024-2026\Models\100126V1\2_Classifier\batnet-lstm\normalization_stats.json"

# ============================================================
# PARALLEL PROCESSING
# ============================================================
NUM_WORKERS = 24

# Model configuration
# ============================================================
# LSTM MODEL CONFIGURATION
# ============================================================
BATCH_SIZE = 32
HIDDEN_SIZE = 128
NUM_LAYERS = 2
LEARNING_RATE = 0.001
DROPOUT = 0.2
ATTENTION_DROPOUT = 0.2
TARGET_REPLICATION_ALPHA = 0.0
NUM_EPOCHS = 200
MAX_SEQUENCE_LENGTH = 300
VALIDATION_RATIO = 0.15
PATIENCE = 20
RANDOM_SEED = 7
WEIGHT_DECAY = 1e-5

FEATURE_COLUMNS = [
    f"{column}_normalized"
    for column in COLUMNS
]
