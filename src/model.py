import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from src.config import (
    MODELS_DIR, MODEL_FILENAME, TEST_SIZE, RANDOM_STATE,
    N_ESTIMATORS, MAX_DEPTH, LEARNING_RATE, SUBSAMPLE, COLSAMPLE_BYTREE
)
from src.preprocess import get_preprocessor

def build_pipeline(scale_pos_weight):
    """Constructs the ML pipeline."""
    preprocessor = get_preprocessor()
    
    classifier = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        n_estimators=N_ESTIMATORS,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        verbosity=0
    )
    
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Trains model, evaluates, and saves artifacts."""
    
    # Calculate class weight from TRAIN ONLY to avoid leakage
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    pipeline = build_pipeline(scale_pos_weight)
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # Metrics
    auc = roc_auc_score(y_test, y_pred_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Save Model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / MODEL_FILENAME
    
    # XGBoost native save is better for interoperability than pickle
    booster = pipeline.named_steps['classifier'].get_booster()
    booster.save_model(str(model_path))
    
    # Save metadata for inference
    meta = {
        "feature_names": list(pipeline.named_steps['preprocessor'].get_feature_names_out()),
        "classes": [0, 1]
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(meta, f)
        
    return pipeline, {"auc": auc, "report": report}