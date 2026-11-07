from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports" / "figures"

CSV_FILENAME = "term-deposit-marketing-2020.csv"
MODEL_FILENAME = "xgb_term_deposit.json"

# Columns
TARGET_COL = "y"
NUMERIC_COLS = ["age", "balance", "day", "campaign"] # Excluded 'duration' for pre-call prediction
CATEGORICAL_COLS = [
    "job", "marital", "education", "default", 
    "housing", "loan", "contact", "month"
]

# Hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 200
MAX_DEPTH = 6
LEARNING_RATE = 0.1
SUBSAMPLE = 0.8
COLSAMPLE_BYTREE = 0.8