# model_service/tests/production_tests.py
import sys
import os
import pytest
import json
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model_service.utils import format_timestamp
from model_service.model_manager import (
    FEATURE_COLS,
    load_model,
    _train_and_register,
    _train_and_register_global,
    _train_and_register_arima,
    MODELS_DIR,
    REGISTRY_PATH,
    registry_manager
)
from model_service.app import app
from fastapi.testclient import TestClient

TEST_TICKERS = ["AAPL", "MSFT", "GOOGL"]
client = TestClient(app)


# ---------------------------
# Session-level setup
# ---------------------------
@pytest.fixture(scope="session", autouse=True)
def prepare_models_and_registry():
    """Prepare models and registry for all production tests."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Initialize empty registry if not exists
    if not os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "w") as f:
            json.dump({}, f)

    # Pretrain global models
    for model_name in ["rf", "xgb"]:
        _train_and_register_global(model_name, TEST_TICKERS[0])

    # Pretrain single-ticker models
    for model_name in ["rf", "xgb"]:
        _train_and_register(model_name, TEST_TICKERS[0])

    # Pretrain ARIMA
    _train_and_register_arima(TEST_TICKERS[0])

    yield


# ---------------------------
# Fixture to make registry stale
# ---------------------------
@pytest.fixture
def stale_registry():
    """Force all registry entries to be stale by backdating last_trained_at."""
    reg = registry_manager.load(force=True)
    stale_time = format_timestamp(datetime.now(timezone.utc) - timedelta(days=10))
    print(f"[stale_registry] Setting all models last_trained_at to: {stale_time}")

    for ticker, models in reg.items():
        for model_name, model_info in models.items():
            registry_manager.update_model(
                ticker=ticker,
                model_name=model_name,
                model_path=model_info["model_path"],
                feature_cols=model_info.get("feature_cols", []),
                tickers_included=model_info.get("tickers_included") if ticker == "global" else None,
                last_trained=stale_time
            )

    # Force reload cache
    registry_manager.load(force=True)
    print("[stale_registry] Registry after making stale:")
    print(json.dumps(registry_manager._cache, indent=2))

    yield


# ===========================
# Phase 1: Non-stale tests
# ===========================
@pytest.mark.parametrize("model_name", ["rf", "xgb"])
def test_global_model_keep(model_name):
    """Global models should not retrain if not stale."""
    info_before = registry_manager.get_model_info("global", model_name)
    model = load_model(model_name, TEST_TICKERS[0], retrain_days=7, use_global=True)
    info_after = registry_manager.get_model_info("global", model_name)

    assert model is not None
    # Non-stale models must retain last_trained_at
    assert info_before["last_trained_at"] == info_after["last_trained_at"]


@pytest.mark.parametrize("model_name", ["rf", "xgb"])
def test_single_model_keep(model_name):
    """Single-ticker models should not retrain if not stale."""
    info_before = registry_manager.get_model_info(TEST_TICKERS[0], model_name)
    model = load_model(model_name, TEST_TICKERS[0], retrain_days=7, use_global=False)
    info_after = registry_manager.get_model_info(TEST_TICKERS[0], model_name)

    assert model is not None
    assert info_before["last_trained_at"] == info_after["last_trained_at"]


def test_arima_keep():
    """ARIMA should not retrain if not stale."""
    info_before = registry_manager.get_model_info(TEST_TICKERS[0], "arima")
    model = load_model("arima", TEST_TICKERS[0], retrain_days=7)
    info_after = registry_manager.get_model_info(TEST_TICKERS[0], "arima")

    assert model is not None
    assert info_before["last_trained_at"] == info_after["last_trained_at"]


# ===========================
# Phase 2: Stale tests
# ===========================
@pytest.mark.parametrize("model_name", ["rf", "xgb"])
def test_global_model_retrain_if_stale(model_name, stale_registry):
    """Global models should retrain if stale."""
    info_before = registry_manager.get_model_info("global", model_name)
    model = load_model(model_name, TEST_TICKERS[0], retrain_days=7, use_global=True)
    info_after = registry_manager.get_model_info("global", model_name)

    assert model is not None
    assert info_before["last_trained_at"] != info_after["last_trained_at"]


@pytest.mark.parametrize("model_name", ["rf", "xgb"])
def test_single_model_retrain_if_stale(model_name, stale_registry):
    """Single-ticker models should retrain if stale."""
    info_before = registry_manager.get_model_info(TEST_TICKERS[0], model_name)
    model = load_model(model_name, TEST_TICKERS[0], retrain_days=7, use_global=False)
    info_after = registry_manager.get_model_info(TEST_TICKERS[0], model_name)

    assert model is not None
    assert info_before["last_trained_at"] != info_after["last_trained_at"]


def test_arima_retrain_if_stale(stale_registry):
    """ARIMA should retrain if stale."""
    info_before = registry_manager.get_model_info(TEST_TICKERS[0], "arima")
    model = load_model("arima", TEST_TICKERS[0], retrain_days=7)
    info_after = registry_manager.get_model_info(TEST_TICKERS[0], "arima")

    assert model is not None
    assert info_before["last_trained_at"] != info_after["last_trained_at"]
