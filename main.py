from sklearn.model_selection import train_test_split
from src.preprocess import load_data, prepare_features
from src.model import train_and_evaluate
from src.interpretability import generate_plots
from src.config import TEST_SIZE, RANDOM_STATE

def main():
    print("--- Loading Data ---")
    df = load_data()
    
    print("--- Preparing Features ---")
    X, y = prepare_features(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    print("--- Training Model ---")
    pipeline, metrics = train_and_evaluate(X_train, y_train, X_test, y_test)
    
    print("--- Generating Interpretability Reports ---")
    # Get feature names after preprocessing for plotting labels
    preprocessor = pipeline.named_steps['preprocessor']
    feature_names = list(preprocessor.get_feature_names_out())
    
    generate_plots(pipeline, X_test, y_test, feature_names)
    
    print("--- Done. Check reports/figures and models/ directories ---")

if __name__ == "__main__":
    main()