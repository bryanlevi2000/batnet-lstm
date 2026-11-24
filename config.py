

# ============================================================
# Global settings
# ============================================================

# Recordings are found recursively. WAV files, .zc files and files ending in "#" (Anabat ZC files) are all read.
INPUT_DIR = r"C:\Users\Students\Box\BatNET 2024-2026\Bat Recordings (Call Library)\All Calls by Dataset (Original, All files)\Bat Call Library - Kim & Chris\EPFU"
NUM_WORKERS = 8

# ============================================================
# Stage 1: ZC Extraction
# ============================================================

OUTPUT_DIR = r"C:\Users\Students\Desktop\Working\zc"

# Output mode - specify "individual" to have one CSV per pulse,
# or "combined" to have one CSV per input file
OUTPUT_MODE = "combined"  # individual or combined

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
METRIC_OUTPUT_FILE = r"C:\Users\Students\Box\BatNET 2024-2026\Bryan Files\Working\111626V1_training_data\mytluc_metrics_output.csv"

# ============================================================
# Stage 2: Preprocessing
# ============================================================

PULSE_FILTERED_REJECTION_FILE = r"C:\Users\Students\Box\BatNET 2024-2026\Models\111326V1\2_Classifier\pulse_rejection_summary.csv"
PCHIP_TIME_INTERVAL = 100  # microseconds

# preprocess/pipeline.py: files that could not be read or processed (columns: file, error),
# and files that were read but produced zero pulses (column: file)
PREPROCESS_FAILED_FILES_PATH = r"C:\Users\Students\Desktop\Working\zc\preprocess_failed_files.csv"
PREPROCESS_NO_PULSE_FILES_PATH = r"C:\Users\Students\Desktop\Working\zc\preprocess_no_pulse_files.csv"

# preprocess/get_normalization_stats.py: same two lists
STATS_FAILED_FILES_PATH = r"C:\Users\Students\Desktop\Working\zc\stats_failed_files.csv"
STATS_NO_PULSE_FILES_PATH = r"C:\Users\Students\Desktop\Working\zc\stats_no_pulse_files.csv"

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
SHARD_PARENT_DIR = r"C:\Users\Students\Desktop\Working\model_training_111326V1\shards"
VAL_RATIO = 0.15

# build_shards.py: files that could not be read or processed (columns: file, error),
# and files that were read but produced zero pulses (column: file)
SHARD_FAILED_FILES_PATH = r"C:\Users\Students\Desktop\Working\model_training_111326V1\shards\failed_files.csv"
SHARD_NO_PULSE_FILES_PATH = r"C:\Users\Students\Desktop\Working\model_training_111326V1\shards\no_pulse_files.csv"

# ============================================================
# Stage 4: Classifier
# ============================================================

TRAIN_SHARD_DIR = r"C:\Users\Students\Desktop\Working\model_training_111326V1\shards" # Parent directory of where the training shards are located
CLASSIFIER_OUTPUT_DIR = r"C:\Users\Students\Box\BatNET 2024-2026\Models\112026V1\2_Classifier" # Folder where classifier and log are written

FEATURE_COLUMNS = ["freq_hz"] # Subset of NORMALIZATION_COLUMNS to use for training
BATCH_SIZE = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 4
LEARNING_RATE = 0.001
DROPOUT = 0.2
ATTENTION_DROPOUT = 0.2
TARGET_REPLICATION_ALPHA = 0.0 # 0 disables target replication
NUM_EPOCHS = 200
MAX_SEQUENCE_LENGTH = 300 # Longer samples are truncated to the first N steps
PATIENCE = 20
WEIGHT_DECAY = 1e-3
RAM_RESERVE_GB = 4

# Learning rate schedule. "none" keeps LEARNING_RATE constant for the whole run.
# "plateau" multiplies the learning rate by LR_FACTOR each time the validation loss has gone
# LR_PATIENCE + 1 epochs in a row without a new best. Keep PATIENCE (early stopping) well above
# LR_PATIENCE + 1, or training stops before the reductions can help.
LR_SCHEDULER = "plateau" # "none" or "plateau"
LR_FACTOR = 0.7 # Plateau: each reduction multiplies the learning rate by this (between 0 and 1)
LR_PATIENCE = 5 # Plateau: reduce after this many + 1 epochs without a new best validation loss
LR_MIN = 1e-5 # Plateau: the learning rate is never reduced below this

# --------------------- INVESTIGATION (temporary, see INVESTIGATION_CHANGES.md) ---------------------

SEED = 1 # None disables seeding; set per epoch as SEED + epoch
GRAD_CLIP_NORM = 1.0 # Max total gradient norm; None disables clipping (experiment only)
DIAG_LOG_EVERY = 100 # Write a step_diagnostics.csv row every N optimizer steps
SAVE_EVERY_EPOCH = True # Save epochs/epoch_XXX.pt (model + optimizer) after every epoch
RESUME_FROM = None # Path to an epochs/epoch_XXX.pt to branch from; None starts fresh

# --------------------- Inference ---------------------

TEST_DIR = r"C:\Users\Students\Box\BatNET 2024-2026\Bat Recordings (Call Library)\Subsets for Development\Test Sets\kim-chris_usfws_12" # WAV and ZC files to predict; each file's parent folder is its ground truth
CHECKPOINT_PATH = r"C:\Users\Students\Box\BatNET 2024-2026\Models\112026V1\2_Classifier\run_20261120_171706\best_model.pt" # Model used by inference.py
PREDICTIONS_PATH = r"C:\Users\Students\Box\BatNET 2024-2026\Models\112026V2\4_Outputs\predictions.csv"
FAILED_FILES_PATH = r"C:\Users\Students\Box\BatNET 2024-2026\Models\112026V2\4_Outputs\failed_files.csv" # Files that could not be processed
NO_PULSE_FILES_PATH = r"C:\Users\Students\Box\BatNET 2024-2026\Models\112026V2\4_Outputs\no_pulse_files.csv" # Files that were read but gave no predictions

# validation_inference.py: validation files listed in TRAIN_SHARD_DIR, found under INPUT_DIR
VAL_PREDICTIONS_PATH = r"C:\Users\Students\Box\BatNET 2024-2026\Models\112026V2\4_Outputs\val_predictions.csv"
VAL_FAILED_FILES_PATH = r"C:\Users\Students\Box\BatNET 2024-2026\Models\112026V2\4_Outputs\val_failed_files.csv"
VAL_NO_PULSE_FILES_PATH = r"C:\Users\Students\Box\BatNET 2024-2026\Models\112026V2\4_Outputs\val_no_pulse_files.csv"
