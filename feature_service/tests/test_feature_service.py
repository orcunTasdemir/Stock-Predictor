import sys
import os
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app
from fastapi.testclient import TestClient

TEST_TICKER = "AAPL"
client = TestClient(app)

def test_feature_endpoint():
    """
    Smoke test for feature service /features endpoint
    """
    # Replace with minimal raw data required by your feature service
    raw_data = [{"Date": "2025-01-01", "Open": 100, "High": 110, "Low": 90, "Close": 105, "Volume": 1000}]

    response = client.post("/features", json={"ticker": TEST_TICKER, "raw_data": raw_data})
    
    data = response.json()
    print("[Feature Service] Features response:", data)
    assert "features" in data
    assert response.status_code == 200


