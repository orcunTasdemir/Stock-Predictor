from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import numpy as np

app = FastAPI(title="Feature Service")

# Pydantic model for input data validation
class RawStockData(BaseModel):
    ticker: str
    raw_data: List[Dict] # list of OHLCV dictionaries
    
# Helper function for feature engineering
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    
    # `Ensure the data is sorted by date
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    
    # Simple Moving Average (SMA)
    df['SMA_3'] = df['Close'].rolling(window=3).mean()
    
    # Exponential Moving Average (EMA)
    df['EMA_3'] = df['Close'].ewm(span=3, adjust=False).mean()
    
    # Daily returns
    df['Returns'] = df['Close'].pct_change()
    
    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    rs = roll_up / roll_down
    df['RSI_14'] = 100.0 - (100.0 / (1 + rs))
    
    # Fill NaN values for first few rows for some indicators
    df.fillna(0, inplace=True)
    return df


@app.post("/features")
def generate_features(data: RawStockData):
          try:
                df = pd.DataFrame(data.raw_data)
                df_feature = add_technical_indicators(df)
                
                # Return as list of dicts
                return {
                    "ticker": data.ticker, "features": df_feature.to_dict(orient='records')}
          except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))