# services/cluster_service.py
import joblib
import pandas as pd

# Model cache to avoid reloading
_model_cache = None


def _load_model():
    """Lazy load clustering model on first use"""
    global _model_cache
    if _model_cache is None:
        print("Loading K-means clustering model...")
        _model_cache = joblib.load("models/cluster/unsupervise_algokmeans_pipeline.pkl")
        print("Clustering model loaded successfully!")
    return _model_cache


def predict_cluster(data):
    """
    Predict the cluster for a social media post.

    Args:
        data: Dictionary containing social media engagement metrics

    Returns:
        int: Cluster ID (0 or 1)
    """
    model = _load_model()
    df = pd.DataFrame([data])
    pred = model.predict(df)
    return int(pred[0])