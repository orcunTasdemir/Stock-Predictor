# model_service/models_impl/rf_model.py
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_rf_model(X, y, model_path: str):
    """Train and save a RandomForest model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump({"model": model}, model_path)
    return model
