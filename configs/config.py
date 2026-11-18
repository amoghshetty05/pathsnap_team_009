"""Central configuration for PathSnap. All paths are relative to the repo root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "raw"
TRAIN_DIR = DATA_DIR / "train"
STUDENT_TEST_DIR = DATA_DIR / "student_test"
PROCESSED_DIR = ROOT / "data" / "processed"

RESULTS_DIR = ROOT / "results"
ARTIFACT_DIR = ROOT / "models" / "artifacts"

SEED = 42

# Candidate generation (metres). Chosen from candidate-miss-rate analysis on TRAIN only.
CANDIDATE_RADIUS_M = 50.0
CANDIDATE_FALLBACK_RADIUS_M = 150.0  # used only when nothing lies within CANDIDATE_RADIUS_M
MAX_CANDIDATES = 12
DENSIFY_STEP_M = 10.0

# Points whose aligned reference arc is further than this are treated as unreliable labels
# (reference route / GPS disagreement) and excluded from training and scoring.
LABEL_MAX_DIST_M = 60.0

# Training-row subsampling: consecutive 1 Hz points are near-duplicates.
TRAIN_POINT_STRIDE = 3

# Window used for context features (points on each side).
CONTEXT_WINDOW = 5

CV_FOLDS = 5
