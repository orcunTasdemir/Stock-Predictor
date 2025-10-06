# model_service/utils.py
from datetime import datetime, timezone, timedelta
import yfinance as yf
import pandas as pd
import os
import json

# -----------------------
# Feature Columns
# -----------------------
FEATURE_COLS = ["SMA_3", "EMA_3", "Returns", "RSI_14", "Volume"]

# -----------------------
# Timestamp Utilities
# -----------------------
def format_timestamp(dt: datetime) -> str:
    """Format datetime as UTC ISO8601 with 'Z' suffix."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def parse_timestamp(ts: str) -> datetime:
    """Parse timestamp string (Z-format) back into a timezone-aware datetime."""
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

def is_stale(ts: str, days: int) -> bool:
    """Check if a stored timestamp is older than `days`."""
    last_trained = parse_timestamp(ts)
    print(f'last trained at (parsed): {last_trained}')
    return datetime.now(timezone.utc) - last_trained > timedelta(days=days)

# -----------------------
# Stock Data Fetching
# -----------------------
def fetch_stock_data(ticker: str, period="1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
    df["Returns"] = df["Close"].pct_change()
    df["SMA_3"] = df["Close"].rolling(window=3).mean()
    df["EMA_3"] = df["Close"].ewm(span=3, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    df = df.dropna()
    return df
