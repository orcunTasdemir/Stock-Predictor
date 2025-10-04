# smoke_test.py
"""
Smoke test suite for the stock predictor backend.
Covers:
- Gateway connectivity
- Model service predictions (RF, XGB, ARIMA)
- Auto-training checks
"""

import os
import time
import requests

# Import our RegistryManager for registry checks
from model_service.registry_manager import RegistryManager

# -------------------------
# Paths & URLs
# -------------------------
BASE_DIR = os.path.dirname(__file__)  # root dir (where this file lives)
REGISTRY_PATH = os.path.join(BASE_DIR, "model_service", "registry.json")

GATEWAY_URL = "http://127.0.0.1:8000"
MODEL_SERVICE_URL = "http://127.0.0.1:8003"
TEST_TICKER = "AAPL"

# Single shared registry instance
registry_manager = RegistryManager(REGISTRY_PATH)


# -------------------------
# Gateway smoke test
# -------------------------
def test_gateway():
    print("==== Testing Gateway ====")
    try:
        # Correct request: ticker in path, model as query param
        resp = requests.get(f"{GATEWAY_URL}/predict/{TEST_TICKER}", params={"model": "rf"})
        resp.raise_for_status()
        print("Gateway response:", resp.json())
    except Exception as e:
        print("Gateway test failed:", e)


# -------------------------
# Model service test
# -------------------------
def test_model_service(model_name: str):
    """
    Hit the /predict endpoint of the model service with fake features.
    """
    print(f"==== Testing Model Service: {model_name.upper()} ====")

    # Dummy features for the last few days
    payload = {
        "ticker": TEST_TICKER,
        "features": [
            {"Date": "2025-09-30", "Open": 170, "High": 175, "Low": 169, "Close": 172, "Volume": 100000},
            {"Date": "2025-10-01", "Open": 172, "High": 178, "Low": 171, "Close": 174, "Volume": 120000},
            {"Date": "2025-10-02", "Open": 174, "High": 180, "Low": 173, "Close": 176, "Volume": 110000}
        ]
    }

    try:
        resp = requests.post(f"{MODEL_SERVICE_URL}/predict/{model_name}", json=payload)
        print(f"[Model Service] {model_name.upper()} response:")
        print(resp.json())
    except Exception as e:
        print(f"Model service {model_name.upper()} failed:", e)


# -------------------------
# Auto-training test
# -------------------------
def test_auto_training(model_name: str):
    """
    Check registry before and after prediction to see if auto-training happened.
    """

    # --- Before ---
    info_before = registry_manager.get_model_info(TEST_TICKER, model_name)
    exists_before = info_before is not None
    before_ts = info_before["last_trained_at"] if info_before else None

    print(f"[Auto-Training] Before {model_name.upper()} exists:", exists_before)
    if before_ts:
        print(f"[Auto-Training] Last trained at before: {before_ts}")

    # --- Trigger prediction (may cause auto-training) ---
    test_model_service(model_name)
    time.sleep(1)  # allow training to complete

    # --- After ---
    info_after = registry_manager.get_model_info(TEST_TICKER, model_name)
    exists_after = info_after is not None
    after_ts = info_after["last_trained_at"] if info_after else None

    print(f"[Auto-Training] After {model_name.upper()} exists:", exists_after)
    if after_ts:
        print(f"[Auto-Training] Last trained at after: {after_ts}")

    # --- Result ---
    if not exists_before and exists_after:
        print(f"✅ Auto-training triggered: new {model_name.upper()} created!")
    elif exists_before and before_ts != after_ts:
        print(f"✅ Auto-training triggered: stale {model_name.upper()} retrained!")
    elif exists_before and before_ts == after_ts:
        print(f"ℹ️ {model_name.upper()} already existed and was fresh; no retraining needed.")
    else:
        print(f"❌ Auto-training did NOT trigger for {model_name.upper()}.")


# -------------------------
# Run all tests
# -------------------------
if __name__ == "__main__":
    test_gateway()
    test_model_service("arima")
    test_auto_training("rf")
    test_auto_training("xgb")
