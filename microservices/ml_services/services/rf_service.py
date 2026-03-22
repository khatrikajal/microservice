import joblib
import pandas as pd

# Model cache to avoid reloading
_model_cache = None


def _load_model():
    """Lazy load Random Forest model on first use"""
    global _model_cache
    if _model_cache is None:
        print("Loading Random Forest model...")
        _model_cache = joblib.load("models/random_forest/random_forest_pipeline.pkl")
        print("Random Forest model loaded successfully!")
    return _model_cache


def predict_rf(data):
    """
    Predict car acceptability using Random Forest classifier.

    Args:
        data: Dictionary containing car attributes

    Returns:
        str: Prediction category (unacc, acc, good, vgood)
    """
    model = _load_model()
    df = pd.DataFrame([data])
    pred = model.predict(df)

    # Return the prediction as string
    return str(pred[0])