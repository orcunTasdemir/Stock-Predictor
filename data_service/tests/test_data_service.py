import sys
import os
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app
from fastapi.testclient import TestClient

TEST_TICKER = "AAPL"
client = TestClient(app)

def test_data_endpoint():
    """
    Smoke test for data service /stock/<ticker>
    """
    response = client.get(f"/stock/{TEST_TICKER}")
    assert response.status_code == 200
    data = response.json()
    print("[Data Service] Stock response:", data)
    assert isinstance(data, list)  # assuming your endpoint returns list of stock data
