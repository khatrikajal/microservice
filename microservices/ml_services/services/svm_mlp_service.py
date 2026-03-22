import joblib
import pandas as pd
import numpy as np
import sys


# recreate training functions used in pipeline
def extract_date(X):
    X = X.copy()
    if 'Date' in X.columns:
        X['Date'] = pd.to_datetime(X['Date'])
        X['Year'] = X['Date'].dt.year
        X['Month'] = X['Date'].dt.month
        X['Day'] = X['Date'].dt.day
        X = X.drop(columns=['Date'])
    return X


def cap_outliers(X):
    X = X.copy()

    caps = {
        'Rainfall': 3.2,
        'Evaporation': 21.8,
        'WindSpeed9am': 55,
        'WindSpeed3pm': 57
    }

    for col, cap in caps.items():
        if col in X.columns:
            X[col] = np.where(X[col] > cap, cap, X[col])

    return X


# Inject functions into __main__ namespace so pickle can find them
sys.modules['__main__'].extract_date = extract_date
sys.modules['__main__'].cap_outliers = cap_outliers

# Model cache to avoid reloading
_model_cache = None


def _load_models():
    """Lazy load models on first use"""
    global _model_cache
    if _model_cache is None:
        print("Loading SVM/MLP models...")
        _model_cache = joblib.load("models/svm_mlp/all_models.pkl")
        print(f"Models loaded: {list(_model_cache.keys())}")
    return _model_cache


def predict_svm_mlp(data, model_name: str = "svm"):
    """
    Predict using either SVM or MLP model.

    Args:
        data: Dictionary containing weather features
        model_name: Either 'svm' or 'mlp' (default: 'svm')

    Returns:
        int: Prediction (0 or 1)
    """
    all_models = _load_models()

    if model_name not in all_models:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(all_models.keys())}")

    # Get the selected model
    model = all_models[model_name]

    # Convert data to DataFrame and predict
    df = pd.DataFrame([data])
    pred = model.predict(df)

    # Convert prediction to binary (Yes -> 1, No -> 0)
    result = pred[0]
    if isinstance(result, str):
        return 1 if result.lower() in ['yes', '1'] else 0
    else:
        return int(result)