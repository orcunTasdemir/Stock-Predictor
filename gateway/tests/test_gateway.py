import sys
import os
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app
from fastapi.testclient import TestClient

TEST_TICKER = "AAPL"
client = TestClient(app)

def test_gateway_predict():
    """
    Smoke test for gateway prediction endpoint
    """
    response = client.get(f"/predict/{TEST_TICKER}?model=rf")
    assert response.status_code == 200
    data = response.json()
    print("[Gateway] Prediction response:", data)
    assert "prediction" in data
