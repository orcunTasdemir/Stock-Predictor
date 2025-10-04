# model_service/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd

from .model_manager import load_model, _train_and_register, registry_manager, FEATURE_COLS

app = FastAPI(title="Model Service")

# ---------------------------
# Pydantic model for input
# ---------------------------
class FeaturesData(BaseModel):
    ticker: str
    features: List[Dict]  # list of feature dictionaries per date


# ---------------------------
# Prediction endpoint
# ---------------------------
@app.post("/predict/{model_name}")
def predict_model(model_name: str, data: FeaturesData):
    """
    Dynamic model prediction endpoint for ARIMA, RF, XGB, etc.
    """
    
    print("Incoming model_name:", model_name)
    print("Incoming features:", data.features[:2])  # just first 2 rows

    try:
        # Convert incoming features to DataFrame
        df = pd.DataFrame(data.features)
        print("DataFrame columns:", df.columns.tolist())
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)

        model_name = model_name.lower()

        # ---------------------------
        # ARIMA model
        # ---------------------------
        if model_name == "arima":
            if "Close" not in df.columns:
                raise HTTPException(status_code=400, detail="ARIMA requires 'Close' column")
            # Just load ARIMA from registry (will auto-train if missing or stale)
            model = load_model("arima", data.ticker)
            forecast = model.forecast(steps=1)
            next_day_price = forecast.iloc[0]

        # ---------------------------
        # ML models: RF, XGB, etc.
        # ---------------------------
        else:
            missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
            if missing_cols:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing feature columns for {model_name}: {missing_cols}"
                )

            X = df[FEATURE_COLS].values
            model = load_model(model_name, data.ticker)
            next_day_price = float(model.predict([X[-1]])[0])

        return {
            "ticker": data.ticker,
            "model": model_name,
            "prediction": round(float(next_day_price), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------
# Manual train/retrain endpoint
# ---------------------------
@app.post("/train/{model_name}/{ticker}")
def train_model(model_name: str, ticker: str):
    """
    Manually train or retrain a model for a given ticker.
    """
    try:
        model = _train_and_register(model_name.lower(), ticker)
        info = registry_manager.get_model_info(ticker, model_name.lower())
        return {
            "ticker": ticker,
            "model": model_name.lower(),
            "status": "trained",
            "trained_at": info["last_trained_at"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")
