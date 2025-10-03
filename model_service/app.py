from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import os


# Load RF Model
rf_metadata_path = os.path.join(os.path.dirname(__file__), "rf_model.pkl")
rf_metadata = joblib.load(rf_metadata_path)
rf_model = rf_metadata["model"]
rf_feature_cols = rf_metadata["feature_cols"]


app = FastAPI(title="Model Service")

# Pydantic model for input
class FeaturesData(BaseModel):
    ticker: str
    features: List[Dict]  # list of feature dictionaries per date
    



@app.post("/predict/{model_name}")
def predict_model(model_name: str, data: FeaturesData):
    """
    Dynamic model prediction endpoint
    """
    
    try:
        df = pd.DataFrame(data.features)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        
        # Preparing features for the ML models
        X  = df.drop(columns=['Date', 'Close'], errors='ignore').values
        
        ## MODEL SELECTION ##
        if model_name.lower() == 'arima':
            # Use Close column as the time series
            close_series = df['Close']
            # Minimal ARIMA example order=(1,1,1)
            model = ARIMA(close_series, order=(1, 1, 1))
            model_fit = model.fit()
            # Forecast the next value
            forecast = model_fit.forecast(steps=1)
            next_day_price = forecast.iloc[0]
            
        elif model_name.lower() in ['rf', 'randomforest', 'random_forest']:
            try:
                # Ensure df has all required columns
                missing_cols = [col for col in rf_feature_cols if col not in df.columns]
                if missing_cols:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing feature columns for RF: {missing_cols}"
                    )

                # Extract features in the same order as during training
                X = df[rf_feature_cols].values

                # Use last row for prediction
                next_day_price = float(rf_model.predict([X[-1]]))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"RF prediction error: {e}")
                
        elif model_name.lower() in ['xgboost', 'xgb']:
            n_features = X.shape[1]
            if n_features == 0:
                next_day_price = float(df['Close'].iloc[-1])  # fallback to last close price
            else:
                xgb_model = XGBRegressor()
                xgb_model.fit(np.arange(10*n_features).reshape(10, n_features), np.arange(10))
                next_day_price = float(xgb_model.predict([X[-1]]))        
                   
        else:
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' not supported")
        
        return {"ticker": data.ticker, 
                "model": model_name.lower(),
                "prediction": round(float(next_day_price), 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))