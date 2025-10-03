# train_rf_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# -----------------------------
# Prepare training data
# -----------------------------
# Example: simulate a feature DataFrame
# In real life, replace this with your Feature service output
feature_cols = ['SMA_3', 'EMA_3', 'Returns', 'RSI_14', 'Volume']
n_samples = 100

# Random features for demonstration
X_train = np.random.rand(n_samples, len(feature_cols))

# Random target (next-day Close price)
y_train = np.random.rand(n_samples)

# -----------------------------
# 2️⃣ Train RandomForest
# -----------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# -----------------------------
# 3️⃣ Save model + feature columns
# -----------------------------
model_metadata = {
    "model": rf_model,
    "feature_cols": feature_cols
}

model_path = os.path.join(os.path.dirname(__file__), "rf_model.pkl")
joblib.dump(model_metadata, model_path)

print(f"RandomForest model saved to {model_path}")
print(f"Feature columns: {feature_cols}")
