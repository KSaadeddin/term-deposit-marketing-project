import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src.config import REPORTS_DIR, NUMERIC_COLS

def generate_plots(pipeline, X_test, y_test, feature_names):
    """Generates global and local interpretability plots."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        import shap
        HAS_SHAP = True
    except ImportError:
        HAS_SHAP = False
        print("SHAP not installed. Using fallback importance.")

    xgb_clf = pipeline.named_steps['classifier']
    
    if HAS_SHAP:
        explainer = shap.TreeExplainer(xgb_clf)
        # Transform X_test to match model input space for SHAP values calculation
        # Note: SHAP TreeExplainer works best on the original tree inputs or transformed dense arrays
        X_processed = pipeline.named_steps['preprocessor'].transform(X_test)
        
        shap_values = explainer.shap_values(X_processed)
        
        # Global Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_processed, feature_names=feature_names, show=False)
        plt.title("Global Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "shap_global_summary.png")
        plt.close()
        
        # Local Force Plot for one positive case
        pos_indices = np.where(y_test.values == 1)[0]
        if len(pos_indices) > 0:
            idx = pos_indices[0]
            plt.figure(figsize=(10, 6))
            shap.force_plot(explainer.expected_value, shap_values[idx,:], X_processed[idx,:], 
                            feature_names=feature_names, matplotlib=True, show=False)
            plt.title(f"Local Explanation for Customer {idx}")
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / "shap_local_example.png")
            plt.close()
            
    else:
        # Fallback: Permutation Importance
        from sklearn.inspection import permutation_importance
        
        perm_imp = permutation_importance(
            estimator=pipeline, 
            X=X_test, 
            y=y_test, 
            n_repeats=10, 
            random_state=42,
            scoring='roc_auc'
        )
        
        imp_series = pd.Series(perm_imp.importances_mean, index=feature_names).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 8))
        imp_series.head(20).plot(kind='barh')
        plt.gca().invert_yaxis()
        plt.title("Permutation Importance (Fallback)")
        plt.xlabel("Mean decrease in ROC AUC")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "permutation_importance_fallback.png")
        plt.close()
        
        print("Saved fallback importance plot.")