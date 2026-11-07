import pandas as pd
import numpy as np
import re
from io import StringIO
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from src.config import (
    DATA_DIR, CSV_FILENAME, TARGET_COL, 
    NUMERIC_COLS, CATEGORICAL_COLS, RANDOM_STATE
)

def repair_glued_csv_text(text):
    """Repairs rows where target 'y' is glued to next row's age."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Fix header glue: campaign,y58 -> campaign,y\n58
    text = re.sub(r"(campaign,y)(?=\d)", r"\1\n", text, flags=re.IGNORECASE)
    # Fix target glue: no44 -> no\n44
    text = re.sub(r"(?<=,)(yes|no)(?=\d)", lambda m: f"{m.group(1)}\n", text, flags=re.IGNORECASE)
    return text

def load_data():
    """Loads and cleans the raw CSV."""
    filepath = DATA_DIR / CSV_FILENAME
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found at {filepath}")

    raw_text = filepath.read_text(encoding="utf-8-sig")
    raw_text = repair_glued_csv_text(raw_text)
    
    df = pd.read_csv(StringIO(raw_text), sep=",", dtype=str, skip_blank_lines=True)
    df.columns = [col.strip() for col in df.columns]
    
    # Ensure target column exists
    if TARGET_COL not in df.columns:
        last_col = df.columns[-1]
        df.rename(columns={last_col: TARGET_COL}, inplace=True)

    # Map target to binary
    df[TARGET_COL] = df[TARGET_COL].str.lower().map({"yes": 1, "no": 0})
    df.dropna(subset=[TARGET_COL], inplace=True)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Convert numerics
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Drop rows with missing critical numerics
    df.dropna(subset=NUMERIC_COLS, inplace=True)

    # Clean categoricals
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("unknown").astype(str).str.lower()

    return df

def get_preprocessor():
    """Returns fitted ColumnTransformer for train data."""
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS)
        ]
    )

def prepare_features(df):
    """Separates features and target."""
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET_COL]
    return X, y