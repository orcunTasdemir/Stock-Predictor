# model_service/models_impl/xgb_model.py
from xgboost import XGBRegressor
import joblib
import os
import numpy as np

def train_xgb_model(X, y, model_path: str):
    """Train and save an XGBoost model."""
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump({"model": model}, model_path)
    return model
