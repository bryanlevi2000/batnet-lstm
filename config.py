

# ============================================================
# Global settings
# ============================================================

INPUT_DIR = r"C:\Users\Students\Box\BatNET 2024-2026\Bat Recordings (Call Library)\Subsets for Development\model_training\111026V1\wav"
NUM_WORKERS = 16

# ============================================================
# Stage 1: ZC Extraction
# ============================================================

OUTPUT_DIR = r"C:\Users\Students\Desktop\Working\model_training_111026V1"

# Output mode - specify "individual" to have one CSV per pulse,
# or "combined" to have one CSV per WAV file
OUTPUT_MODE = "individual"  # individual or combined

# --------------------- Denoising ---------------------

NOISE_FACTOR = 2.0
NOISE_STRIDE = 4

# --------------------- ZC Extraction ---------------------

DIVRAT = 8
HYSTERESIS_PERCENTILE = 85.0

# --------------------- Pulse Detection ---------------------

# Local linearity
LOOKAHEAD_N = 15
LINEARITY_CUTOFF = 0.9

# Greedy walk
TIME_CUTOFF = 500  # microseconds
INC_FREQ_CUTOFF_HZ = 1000
DEC_FREQ_CUTOFF_HZ = 5000
MIN_POINTS = 15
BUFFER_US = 10000

# --------------------- Pulse Cleanup ---------------------

# Tail removal
TAIL_TIME_FRACTION = 0.1
TAIL_DEC_FREQ_CUTOFF_HZ = 1000

# Pulse metrics
MIN_BANDWIDTH_HZ = 1000
MAX_BANDWIDTH_HZ = 120000

MIN_DURATION_US = 100
MAX_DURATION_US = 30000

FMIN_RANGES = (5000, 50000)
FMAX_RANGES = (10000, 130000)

# --------------------- Pulse Metrics ---------------------

METRIC_FC_PROPORTION = 0.33 # Fc can only occur in this last proportion of the call in time
METRIC_KNEE_PROPORTION = 0.1 # Knee cannot occur in this first proportion of the call in time
METRIC_OUTPUT_FILE = r"C:\Users\Students\Desktop\Working\model_training_111026V1\metrics_output.csv"

# ============================================================
# Stage 2: Preprocessing
# ============================================================

PULSE_FILTERED_REJECTION_FILE = r"C:\Users\Students\Desktop\Working\model_training_111026V1\pulse_rejection_summary.csv"
PCHIP_TIME_INTERVAL = 100  # microseconds

# Currently, the possible features we can include are freq_hz, amplitude, slope, position
NORMALIZATION_COLUMNS = [
    "freq_hz",
    "amplitude",
    "slope",
    "position"
]

# ============================================================
# Stage 3: Shard creation
# ============================================================

SHARD_NUM_SAMPLES = 50000 # Maximum number of samples per shard
SHARD_PARENT_DIR = r"C:\Users\Students\Desktop\Working\model_training_111026V1\shards"
VAL_RATIO = 0.15

# ============================================================
# Stage 4: Classifier
# ============================================================

TRAIN_SHARD_DIR = r"C:\Users\Students\Desktop\Working\model_training_111026V1\shards" # Parent directory of where the training shards are located
CLASSIFIER_OUTPUT_DIR = r"C:\Users\Students\Box\BatNET 2024-2026\Models\111026V1\2_Classifier" # Folder where classifier and log are written

FEATURE_COLUMNS = ["freq_hz", "amplitude"] # Subset of NORMALIZATION_COLUMNS to use for training
BATCH_SIZE = 32
HIDDEN_SIZE = 128
NUM_LAYERS = 2
LEARNING_RATE = 0.001
DROPOUT = 0.2
ATTENTION_DROPOUT = 0.2
TARGET_REPLICATION_ALPHA = 0.5 # 0 disables target replication
NUM_EPOCHS = 200
MAX_SEQUENCE_LENGTH = 300 # Longer samples are truncated to the first N steps
PATIENCE = 20
WEIGHT_DECAY = 1e-5
RAM_RESERVE_GB = 4

# --------------------- Inference ---------------------

TEST_DIR = r"C:\Users\Students\Box\BatNET 2024-2026\Bat Recordings (Call Library)\Subsets for Development\Annotation Dataset\Dataset\f\COYTOV" # WAVs to predict; each WAV's parent folder is its ground truth
CHECKPOINT_PATH = r"C:\Users\Students\Box\BatNET 2024-2026\Models\111026V1\2_Classifier\run_20261109_160353\best_model.pt" # Model used by inference.py
PREDICTIONS_PATH = r"C:\Users\Students\Box\BatNET 2024-2026\Models\111026V1\4_Outputs\predictions.csv"
FAILED_FILES_PATH = r"C:\Users\Students\Box\BatNET 2024-2026\Models\111026V1\4_Outputs\failed_files.csv" # WAVs that could not be processed

# validation_inference.py: validation WAVs listed in TRAIN_SHARD_DIR, found under INPUT_DIR
VAL_PREDICTIONS_PATH = r"C:\Users\Students\Box\BatNET 2024-2026\Models\111026V1\4_Outputs\val_predictions.csv"
VAL_FAILED_FILES_PATH = r"C:\Users\Students\Box\BatNET 2024-2026\Models\111026V1\4_Outputs\val_failed_files.csv"
