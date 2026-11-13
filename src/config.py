# Defines shared paths, seed(42) and label threshold (0.5)
# Owner: Brant
# Run: imported
# Reads env var ARTIFACT_DIR
# Writes: Nothing

import os
from pathlib import Path

# read the environment variable
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "./artifacts")).resolve()

# shared constraints (as discussed in README.md)
SEED = 42
THRESHOLD = 0.5
TRAIN_SAMPLE_SIZE = 200000

# path helpers
SPLITS_DIR = ARTIFACT_DIR / "splits"
MODELS_DIR = ARTIFACT_DIR / "models"
PREDS_DIR = ARTIFACT_DIR / "preds"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# create directories
def ensure_dirs():
    for d in (SPLITS_DIR, MODELS_DIR, PREDS_DIR):
        d.mkdir(parents=True, exist_ok=True)
