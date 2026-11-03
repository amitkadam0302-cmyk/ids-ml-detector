"""Central configuration: paths, dataset URLs, random seed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

TRAIN_FILE = RAW_DIR / "KDDTrain+.txt"
TEST_FILE = RAW_DIR / "KDDTest+.txt"
BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"

RANDOM_STATE = 42
