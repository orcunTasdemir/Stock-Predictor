import os
import joblib
import pandas as pd
from .utils import FEATURE_COLS, fetch_stock_data
from .models_impl.rf_model import train_rf_model
from .models_impl.xgb_model import train_xgb_model
from .registry_manager import RegistryManager
from statsmodels.tsa.arima.model import ARIMA

# Directories
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Registry manager
REGISTRY_PATH = os.path.join(BASE_DIR, "registry.json")
registry_manager = RegistryManager(REGISTRY_PATH)

# -----------------------
# Internal train helpers
# -----------------------
def _train_and_register(model_name: str, ticker: str):
    """Train RF or XGB model and register in registry."""
    print(f"[DEBUG] _train_and_register called for {model_name} {ticker}")
    try:
        df = fetch_stock_data(ticker)
        print(f"[DEBUG] Fetched data for {ticker}, shape: {df.shape}")
        X = df[FEATURE_COLS].values
        y = df["Close"].shift(-1).dropna().values
        X = X[:-1]
        print(f"[DEBUG] X shape: {X.shape}, y shape: {y.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to prepare training data: {e}")
        raise

    model_path = os.path.join(MODELS_DIR, f"{model_name}_{ticker}.pkl")
    print(f"[DEBUG] Model path: {model_path}")

    try:
        if model_name == "rf":
            model = train_rf_model(X, y, model_path)
        elif model_name == "xgb":
            model = train_xgb_model(X, y, model_path)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        print(f"[DEBUG] {model_name} model trained successfully")
    except Exception as e:
        print(f"[ERROR] Failed to train {model_name}: {e}")
        raise

    try:
        registry_manager.update_model(ticker, model_name, model_path, FEATURE_COLS)
        print(f"[DEBUG] Registry updated for {ticker} {model_name}")
    except Exception as e:
        print(f"[ERROR] Failed to update registry: {e}")
        raise

    return model


def _train_and_register_arima(ticker: str):
    """Train ARIMA and register in registry."""
    print(f"[DEBUG] _train_and_register_arima called for {ticker}")
    try:
        df = fetch_stock_data(ticker)
        close_series = df["Close"]
        print(f"[DEBUG] Close series length: {len(close_series)}")

        model = ARIMA(close_series, order=(1, 1, 1))
        model_fit = model.fit()
        print(f"[DEBUG] ARIMA model trained successfully")

        model_path = os.path.join(MODELS_DIR, f"arima_{ticker}.pkl")
        joblib.dump({"model": model_fit}, model_path)
        print(f"[DEBUG] ARIMA model saved at {model_path}")

        registry_manager.update_model(ticker, "arima", model_path, ["Close"])
        print(f"[DEBUG] Registry updated for ARIMA {ticker}")
    except Exception as e:
        print(f"[ERROR] ARIMA training failed: {e}")
        raise

    return model_fit


def load_model(model_name: str, ticker: str, retrain_days: int = 7):
    """Load model; auto-train if missing or stale."""
    print(f"[DEBUG] load_model called for {model_name} {ticker}")
    try:
        info = registry_manager.get_model_info(ticker, model_name)
        print(f"[DEBUG] Registry info: {info}")
        
    except Exception as e:
        print(f"[ERROR] Failed to read registry info: {e}")
        raise

    try:
        print("Hello0",(not info))
        if not info:
            print("Hello1" )
            print(f"[ModelManager] Training new {model_name} model for {ticker}")
            if model_name == "arima":
                return _train_and_register_arima(ticker)
            return _train_and_register(model_name, ticker)

        elif registry_manager.is_model_stale(ticker, model_name, retrain_days): 
            print("Hello2")
            print(f"[ModelManager] Retraining stale {model_name} model for {ticker}")
            if model_name == "arima":
                return _train_and_register_arima(ticker)
            return _train_and_register(model_name, ticker)
        else:
            print("Hello3")
            # Load from disk
            print(f"[DEBUG] Loading model from {info['model_path']}")
            loaded = joblib.load(info["model_path"])
            
            if isinstance(loaded, dict) and "model" in loaded:
                model = loaded["model"]
            else:
                model = loaded
            print(f"[DEBUG] Model loaded successfully")
            return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise
























































# def _train_and_register(model_name: str, ticker: str):
#     """Train RF or XGB model and register in registry."""
#     df = fetch_stock_data(ticker)
#     X = df[FEATURE_COLS].values
#     y = df["Close"].shift(-1).dropna().values
#     X = X[:-1]

#     model_path = os.path.join(MODELS_DIR, f"{model_name}_{ticker}.pkl")

#     if model_name == "rf":
#         model = train_rf_model(X, y, model_path)
#     elif model_name == "xgb":
#         model = train_xgb_model(X, y, model_path)
#     else:
#         raise ValueError(f"Unsupported model: {model_name}")

#     registry_manager.update_model(ticker, model_name, model_path, FEATURE_COLS)
#     return model

# def _train_and_register_arima(ticker: str):
#     """Train ARIMA and register in registry."""
#     df = fetch_stock_data(ticker)
#     close_series = df["Close"]

#     # Train ARIMA model (you can tune order later)
#     model = ARIMA(close_series, order=(1, 1, 1))
#     model_fit = model.fit()

#     # Save ARIMA model
#     model_path = os.path.join(MODELS_DIR, f"arima_{ticker}.pkl")
#     joblib.dump({"model": model_fit}, model_path)

#     # Update registry
#     registry_manager.update_model(ticker, "arima", model_path, ["Close"])
#     return model_fit

# # -----------------------
# # Load model (with retrain if stale/missing)
# # -----------------------
# def load_model(model_name: str, ticker: str, retrain_days: int = 7):
#     """Load model; auto-train if missing or stale."""
#     info = registry_manager.get_model_info(ticker, model_name)
#     if not info:
#         print(f"[ModelManager] Training new {model_name} model for {ticker}")
#         if model_name == "arima":
#             return _train_and_register_arima(ticker)
#         return _train_and_register(model_name, ticker)

#     if registry_manager.is_model_stale(ticker, model_name, retrain_days):
#         print(f"[ModelManager] Retraining stale {model_name} model for {ticker}")
#         if model_name == "arima":
#             return _train_and_register_arima(ticker)
#         return _train_and_register(model_name, ticker)

#     # Load from disk
#     return joblib.load(info["model_path"])["model"]
