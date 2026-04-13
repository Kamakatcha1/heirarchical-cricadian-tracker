# ============================================================
# CENTRAL CONFIGURATION -- edit here once, all scripts use it
# ============================================================
# Set your experiment ID, base directory, and dataset ID here.
# Every script imports these values.  If a script needs to
# deviate (e.g. run with a different experiment), set the
# _OVERRIDE dict at the top of that script instead.
# ============================================================

EXPERIMENT_ID   = "F2_1"

# Project root -- the Hierarchical Circadian Tracker folder itself.
# Everything else is derived from this.
# Set HCT_BASE_DIR env var to override (e.g. in Docker).
import os as _os
BASE_DIR        = _os.environ.get("HCT_BASE_DIR",
                    r"/storage1/fs1/bmansfeld/Active/work/shafay/hct")

# Which raw image dataset to use (subfolder name under data/raw/)
DATASET_ID      = "F2_001"

# ============================================================
# Derived paths -- do not edit below this line
# ============================================================
EXPERIMENTS_DIR = _os.path.join(BASE_DIR, "data", "experiments")
EXPERIMENT_DIR  = _os.path.join(EXPERIMENTS_DIR, EXPERIMENT_ID)
RAW_DIR         = _os.path.join(BASE_DIR, "data", "raw", DATASET_ID)
TRAINING_DIR    = _os.path.join(BASE_DIR, "data", "training")

# ============================================================
# Per-step defaults
# These are the values each script uses unless overridden.
# ============================================================

# 01_crop  (blends composite in memory, then interactive rectangle selection)
BLEND_METHOD    = "max"     # "max" or "mean"
EXPECTED_PLANTS = 0         # 0 = no enforcement, or set to expected plant count
MAX_DISPLAY     = 1100      # max window size for rectangle picking
MAX_FRAMES      = 180         # 0 = use all frames, or set to limit (first N)

# 02_annotate  (virtual crops, annotation coordinates only)
IMAGES_PER_FOLDER = 5       # how many images to annotate per plant 
DISPLAY_SCALE     = 2       # zoom factor for annotation window

# 03_masks  (reads experiment JSONs, generates images + masks into data/training/)
TRAINING_EXPERIMENTS = ["F2_2_2",  "F2_2_1"]   # list ALL experiment IDs to include
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
FN_WEIGHT          = 2.0                # weight of false-negative (missed tip) penalty

# 05_measure + 06_export  (multi-experiment support)
MEASURE_EXPERIMENTS = ["F2_1","F2_2_1", "F2_2_2"]                 # list of experiment IDs to process together, e.g. ["F2_2_1", "F2_2_2"]
                                         # empty = just use EXPERIMENT_ID

# 05_measure  (runs model across crops, outputs tip distance graphs)
MODEL_PATH         = ""                  # path to .keras model, empty = auto-find best.keras
                                         # "shared" = use this one model for all experiments
                                         # "auto"   = each experiment uses its own best.keras
                                         # (only matters when MEASURE_EXPERIMENTS has multiple entries)
MODEL_MODE         = "auto"            # "shared" = one model for all, "auto" = each experiment's own
MEASURE_MAX_FRAMES = 134                   # 0 = use all frames, or set to limit (first N)
NUM_TIPS           = 2                   # number of peaks to detect per plant
MIN_DIST           = 20                  # min distance between peaks (model pixels)
INTERVAL_MIN       = 30                  # minutes between frames

# 06_export  (export formatted CSV from tip_distances)
GENOTYPE_NAMES     = {1: "M82", 2: "Penelli", 3: "pimpi", 4: "F2"}  # map genotype number to name
EXCLUDE_PLANTS     = {                   # plants to exclude from export
    "*":      [],                        # exclude from ALL experiments
    "F2_1":   ["g4_2", "g4_3","g4_8", "g4_12", "g4_16"],                        # exclude only from this experiment
    "F2_2_1": ["g4_2", "g4_22", "g4_26"],
    "F2_2_2": ["g4_1", "g4_2", "g4_3", "g4_4", "g4_5", "g4_6", "g4_8", "g4_10", "g4_11", "g4_16", "g4_18", "g4_20", "g4_21", "g4_22", "g4_28", "g4_27", "g4_26", "g4_23"],
    # Format:  "g4_2"  = genotype 4 replicate 2
    #          "g3_*"  = ALL of genotype 3
    #          "g4_2"  = just that one plant
    # Examples:
    #   "*":      ["g3_*"],              # exclude all genotype 3 from every experiment
    #   "*":      ["g4_2", "g4_8"],      # exclude specific plants from all experiments
    #   "F2_2_1": ["g3_*", "g4_14"],     # mix: all g3 + specific g4 from one experiment
}

# 07_summary  (Biodare detrended summary plots)
EXCLUDE_SAMPLES    = []                    # sample numbers to exclude from summary, e.g. [3, 15, 22]
