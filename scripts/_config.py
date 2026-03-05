# ============================================================
# CENTRAL CONFIGURATION -- edit here once, all scripts use it
# ============================================================
# Set your experiment ID, base directory, and dataset ID here.
# Every script imports these values.  If a script needs to
# deviate (e.g. run with a different experiment), set the
# _OVERRIDE dict at the top of that script instead.
# ============================================================

EXPERIMENT_ID   = "3_genotypes_run1"

# Project root -- the Hierarchical Circadian Tracker folder itself.
# Everything else is derived from this.
BASE_DIR        = r"C:\Users\shafa\OneDrive\Desktop\Leaf Project\Hierarchical Circadian Tracker"

# Which raw image dataset to use (subfolder name under data/raw/)
DATASET_ID      = "3_genotype_combined_01"

# ============================================================
# Derived paths -- do not edit below this line
# ============================================================
EXPERIMENTS_DIR = rf"{BASE_DIR}\data\experiments"
EXPERIMENT_DIR  = rf"{EXPERIMENTS_DIR}\{EXPERIMENT_ID}"
RAW_DIR         = rf"{BASE_DIR}\data\raw\{DATASET_ID}"
TRAINING_DIR    = rf"{BASE_DIR}\data\training"

# ============================================================
# Per-step defaults
# These are the values each script uses unless overridden.
# ============================================================

# 01_crop  (blends composite in memory, then interactive rectangle selection)
BLEND_METHOD    = "max"     # "max" or "mean"
EXPECTED_PLANTS = 0         # 0 = no enforcement, or set to expected plant count
MAX_DISPLAY     = 1100      # max window size for rectangle picking
MAX_FRAMES      = 196         # 0 = use all frames, or set to limit (first N)

# 02_annotate  (virtual crops, annotation coordinates only)
IMAGES_PER_FOLDER = 10       # how many images to annotate per plant 
DISPLAY_SCALE     = 3       # zoom factor for annotation window

# 03_masks  (reads experiment JSONs, generates images + masks into data/training/)
TRAINING_EXPERIMENTS = ["3_genotypes_run1"]   # list ALL experiment IDs to include
GENOTYPE_FILTER    = []                  # empty = all genotypes, or e.g. [1, 3] for only those
SIGMA              = 4                    # gaussian kernel sigma for tip heatmaps
# Augmentation
AUGMENT            = True               # set True to enable augmentation
AUGS_PER_IMAGE     = 4                   # augmented copies per annotated image
AUG_MAX_ROTATE     = 15                  # max rotation degrees for affine augmentation
AUG_MIN_SCALE      = 0.9                 # min scale factor for affine augmentation
AUG_MAX_SCALE      = 1.1                 # max scale factor for affine augmentation
AUG_MAX_SHIFT      = 0.05               # max shift as fraction of image size
AUG_HFLIP_PROB     = 0.5                # horizontal flip probability
AUG_VFLIP_PROB     = 0.2                # vertical flip probability
AUG_BRIGHTNESS_ALPHA = (0.9, 1.1)       # brightness multiplier range
AUG_BRIGHTNESS_BETA  = (-10, 10)        # brightness offset range
AUG_SEED           = 0                   # random seed for reproducibility (0 = random)

# 04_train  (trains U-Net from data/training/ images+masks)
IMG_SIZE           = 128                 # images resized to this for training
BATCH_SIZE         = 8
EPOCHS             = 30
LEARNING_RATE      = 1e-4
VAL_SPLIT          = 0.2                 # fraction of base images held for validation
PATIENCE           = 10                  # early stopping patience (epochs)
TRAIN_SEED         = 1337                # seed for train/val split reproducibility
WMSE_ALPHA         = 50.0               # weighted MSE alpha (upweight heatmap pixels)
DICE_WEIGHT        = 0.5                # weight of soft dice in combined loss

# 05_measure  (runs model across crops, outputs tip distance graphs)
MODEL_PATH         = ""                  # path to .keras model, empty = auto-find best.keras in experiment
NUM_TIPS           = 2                   # number of peaks to detect per plant
MIN_DIST           = 20                  # min distance between peaks (model pixels)
INTERVAL_MIN       = 30                  # minutes between frames

# 06_export  (export formatted CSV from tip_distances)
GENOTYPE_NAMES     = {1: "M82", 2: "Penelli", 3: "pimpi"}  # map genotype number to name
EXCLUDE_PLANTS     = ["g1_r04", "g2_r06", "g2_r08", "g2_r011", "g2_12", "g2_15", "g2_18", "g2_19", "g3_17", "g3_16", "g3_13", "g3_15", "g3_14", "g3_10", "g3_2"]                  # labels to exclude, e.g. ["g1_r03", "g2_r01"]

# 07_summary  (Biodare detrended summary plots)
EXCLUDE_SAMPLES    = [1, 2, 3, 5, 8, 9, 12, 17, 18, 20, 22, 23, 25]                    # sample numbers to exclude from summary, e.g. [3, 15, 22]
