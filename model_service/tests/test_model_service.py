# model_service/tests/test_global_models.py
import sys
import os
import pytest
import shutil
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model_service.model_manager import (
    registry_manager,
    FEATURE_COLS
)
from model_service.app import app
from fastapi.testclient import TestClient

TEST_TICKERS = ["AAPL", "MSFT", "GOOGL"]
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
REGISTRY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../registry.json"))

client = TestClient(app)

@pytest.fixture(autouse=True)
def cleanup_registry_and_models():
    """Clean registry and model files before each test and ensure registry exists."""
    if os.path.exists(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Ensure empty registry.json exists
    with open(REGISTRY_PATH, "w") as f:
        f.write("{}")

    yield


# ---------------------------
# Individual model endpoint tests
# ---------------------------
@pytest.mark.parametrize("model_name", ["rf", "xgb"])
def test_individual_model_endpoint(model_name):
    """Train per-ticker model via endpoint and verify prediction works."""
    ticker = TEST_TICKERS[0]

    response = client.post(f"/predict/{model_name}", json={
        "ticker": ticker,
        "features": [{"Date": "2025-01-01", **{col: 0.1 for col in FEATURE_COLS}, "Close": 100}]
    })
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)


# # ---------------------------
# # ARIMA endpoint test
# # ---------------------------
def test_arima_endpoint_single_prediction():
    """Train ARIMA model and predict via endpoint."""
    ticker = TEST_TICKERS[0]
    response = client.post("/predict/arima?use_global=False", json={
        "ticker": ticker,
        "features": [{"Date": "2025-01-01", "Close": 100}]
    })
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], float)


# # ---------------------------
# # Global model endpoint tests (RF/XGB) with parametrize
# # ---------------------------
@pytest.mark.parametrize("model_name", ["rf", "xgb"])
def test_global_model_retrain_endpoint(model_name):
    """Ensure global models retrain when a new ticker is added."""
    ticker1, ticker2 = TEST_TICKERS[:2]

    # First ticker
    response1 = client.post(f"/predict/{model_name}", json={
        "ticker": ticker1,
        "features": [{"Date": "2025-01-01", **{col: 0.1 for col in FEATURE_COLS}, "Close": 100}]
    })
    assert response1.status_code == 200

    info1 = registry_manager.get_model_info("global", model_name)
    assert ticker1 in info1["tickers_included"]

    # Artificially trigger retrain for second ticker by waiting
    time.sleep(1)
    response2 = client.post(f"/predict/{model_name}", json={
        "ticker": ticker2,
        "features": [{"Date": "2025-01-01", **{col: 0.1 for col in FEATURE_COLS}, "Close": 100}]
    })
    assert response2.status_code == 200

    info2 = registry_manager.get_model_info("global", model_name)
    assert ticker2 in info2["tickers_included"]
    assert len(info2["tickers_included"]) == 2
    assert info2["last_trained_at"] != info1["last_trained_at"]


# # ---------------------------
# # ARIMA edge-case tests
# # ---------------------------
def test_arima_short_series_endpoint():
    """ARIMA handles very short series gracefully."""
    ticker = "SHORT"
    df = pd.DataFrame({"Close": [100, 101, 102]}, index=pd.date_range("2025-01-01", periods=3))
    # Patch fetch_stock_data
    with patch("model_service.model_manager.fetch_stock_data", return_value=df):
        response = client.post("/predict/arima?use_global=False", json={
            "ticker": ticker,
            "features": [{"Date": "2025-01-01", "Close": 100}]
        })
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["prediction"], float)
    


def test_arima_with_nans_endpoint():
    """ARIMA handles NaNs in series."""
    ticker = "NANSTOCK"
    df = pd.DataFrame({"Close": [100, np.nan, 102, 103]}, index=pd.date_range("2025-01-01", periods=4))
    # Patch fetch_stock_data
    with patch("model_service.model_manager.fetch_stock_data", return_value=df):
        response = client.post("/predict/arima?use_global=False", json={
            "ticker": ticker,
            "features": [{"Date": "2025-01-01", "Close": 100}]
        })
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["prediction"], float)


def test_predict_batch_endpoint_all_featurecols():
    """Test /predictbatch endpoint using all FEATURE_COLS, one row per ticker."""
    batch_input = []

    for ticker in TEST_TICKERS[:2]:  # test with first 2 tickers
        row = {"Date": "2025-01-01", "Close": 100}
        # dynamically add all feature columns
        row.update({col: 0.1 for col in FEATURE_COLS})
        batch_input.append({"ticker": ticker, "features": [row]})
    print(f"Batch input for {ticker}: {batch_input[-1]}")

    # -------------------- RF/XGB batch prediction --------------------
    response = client.post("/predictbatch/rf", json=batch_input)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    print("Data: ", data)
    for item in data:
        assert "ticker" in item
        assert "predictions" in item
        assert isinstance(item["predictions"], list)
        assert len(item["predictions"]) == 1  # single row per ticker

    # -------------------- ARIMA batch prediction --------------------
    response_arima = client.post("/predictbatch/arima", json=batch_input)
    assert response_arima.status_code == 200
    data_arima = response_arima.json()
    for item in data_arima:
        assert "ticker" in item
        assert "predictions" in item
        assert len(item["predictions"]) == 1  # single row per ticker

@pytest.mark.parametrize("model_name", ["xgb"])
def test_xgb_global_incremental_update(model_name):
    """Ensure XGB global model retrains/increments properly when new tickers are added."""
    ticker1, ticker2 = TEST_TICKERS[:2]

    # First ticker trains initial global model
    response1 = client.post(f"/predict/{model_name}", json={
        "ticker": ticker1,
        "features": [{"Date": "2025-01-01", **{col: 0.1 for col in FEATURE_COLS}, "Close": 100}]
    })
    assert response1.status_code == 200
    info1 = registry_manager.get_model_info("global", model_name)
    assert ticker1 in info1["tickers_included"]
    first_model_path = info1["model_path"]

    # Second ticker triggers incremental update
    response2 = client.post(f"/predict/{model_name}", json={
        "ticker": ticker2,
        "features": [{"Date": "2025-01-01", **{col: 0.2 for col in FEATURE_COLS}, "Close": 101}]
    })
    assert response2.status_code == 200
    info2 = registry_manager.get_model_info("global", model_name)
    
    # Verify that both tickers are included
    assert ticker1 in info2["tickers_included"]
    assert ticker2 in info2["tickers_included"]

    # Confirm global model path unchanged (incremental update uses same file)
    assert info2["model_path"] == first_model_path

def test_predict_batch_xgb():
    """Test /predictbatch/xgb with multiple tickers and feature rows."""
    batch_input = []

    for ticker in TEST_TICKERS[:2]:
        row = {"Date": "2025-01-01", "Close": 100}
        row.update({col: 0.1 for col in FEATURE_COLS})
        batch_input.append({"ticker": ticker, "features": [row]})

    response = client.post("/predictbatch/xgb", json=batch_input)
    assert response.status_code == 200
    data = response.json()
    for item in data:
        assert "ticker" in item
        assert "predictions" in item
        assert isinstance(item["predictions"], list)
        assert len(item["predictions"]) == 1
