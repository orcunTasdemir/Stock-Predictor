
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model_service.utils import FEATURE_COLS
from model_service.registry_manager import RegistryManager
from model_service.app import app
from fastapi.testclient import TestClient
import time
import pytest

# Ensure the model_service package can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

REGISTRY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "registry.json"))
registry_manager = RegistryManager(REGISTRY_PATH)



TEST_TICKER = "AAPL"

# ---------------------------
# Test client for FastAPI
# ---------------------------
client = TestClient(app)

# ---------------------------
# Test prediction endpoint
# ---------------------------
def test_prediction_endpoint():
    """
    Smoke test: check if /predict/<model_name> endpoint returns a valid response
    """
    response = client.post("/predict/rf", json={
        "ticker": TEST_TICKER,
        "features": [{"Date": "2025-01-01", **{col: 0.1 for col in FEATURE_COLS}, "Close": 100}]
    })
    assert response.status_code == 200
    data = response.json()
    print("[Model Service] RF response:", data)
    assert "prediction" in data

# ---------------------------
# Test auto-training logic
# ---------------------------
def test_auto_training():
    """
    Check registry before and after prediction to see if auto-training is triggered
    """
    registry_before = registry_manager.get_model_info(TEST_TICKER, "rf")
    exists_before = registry_before is not None

    print("[Auto-Training] Before RF exists:", exists_before)

    # Trigger prediction
    test_prediction_endpoint()

    time.sleep(2)  # wait for training if triggered

    registry_manager.load(force=True)  # reload registry after prediction
    registry_after = registry_manager.get_model_info(TEST_TICKER, "rf")
    exists_after = registry_after is not None

    print("[Auto-Training] After RF exists:", exists_after)

    if exists_after and not exists_before:
        print("✅ Auto-training triggered for RF!")
    elif exists_after and exists_before:
        print("ℹ️ Model already existed; freshness logic may apply.")
    else:
        print("❌ Auto-training did NOT trigger for RF.")

    assert exists_after  # ensure model exists after prediction
