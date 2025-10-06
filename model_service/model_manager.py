# model_service/model_manager.py
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import logging
from typing import List, Tuple, Dict
from .utils import FEATURE_COLS, fetch_stock_data, format_timestamp, is_stale
from .models_impl.rf_model import train_rf_model
from .models_impl.xgb_model import train_xgb_model
from .registry_manager import RegistryManager
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

    
# -----------------------
# Logging setup
# -----------------------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------
# Directories & Registry
# -----------------------
BASE_DIR = os.path.dirname(__file__)                          # Base directory for this microservice
MODELS_DIR = os.path.join(BASE_DIR, "models")                 # Directory to store trained model pickles
os.makedirs(MODELS_DIR, exist_ok=True)                        # Ensure directory exists
REGISTRY_PATH = os.path.join(BASE_DIR, "registry.json")       # Path to registry JSON file
registry_manager = RegistryManager(REGISTRY_PATH)             # Initialize RegistryManager instance

# -----------------------
# Internal Helpers
# -----------------------
def _load_model_from_path(model_path: str) -> object:
    """Load a model from disk, handling dict-wrapped models from joblib."""
    loaded = joblib.load(model_path)
    return loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded

def _prepare_features_and_target(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix X and target y for RF/XGB:
    - X: feature columns
    - y: next-day 'Close'
    - Drop last row to align shapes
    """
    X = df[FEATURE_COLS].values
    y = df["Close"].shift(-1).values
    return X[:-1], y[:-1]

def _get_arima_start_params(series: pd.Series) -> Tuple[float, float, float]:
    """
    Compute ARIMA(1,1,1) start parameters to avoid non-stationary / invertible warnings.
    Returns: phi1, theta1, intercept
    """
    diff_series = series.diff().dropna()
    if len(diff_series) < 2:
        phi1, theta1 = 0.0, 0.0
    else:
        pacf_vals = pacf(diff_series, nlags=1)
        phi1 = pacf_vals[1].item() if len(pacf_vals) > 1 else 0.0
        theta1 = 0.0
    intercept = diff_series.mean().item()
    return phi1, theta1, intercept


def _stack_features_for_tickers(tickers: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetch and stack features/targets for multiple tickers.
    Returns combined X, y arrays for global training.
    """
    all_X, all_y = [], []
    for t in tickers:
        df_t = fetch_stock_data(t)
        X_t, y_t = _prepare_features_and_target(df_t)
        all_X.append(X_t)
        all_y.append(y_t)
    X_global = np.vstack(all_X)
    y_global = np.hstack(all_y).ravel()
    return X_global, y_global

def _update_ticker_timestamps(tickers: List[str]) -> Dict[str, str]:
    """
    Update timestamps for a list of tickers to current UTC time.
    Returns a dict ticker -> timestamp string
    """
    now = format_timestamp(datetime.now(timezone.utc))
    return {t: now for t in tickers}

# -----------------------
# Train helpers
# -----------------------
def _train_and_register(model_name: str, ticker: str) -> object:
    """
    Train a per-ticker RF/XGB model and register it in registry.
    Always retrains; staleness handled in load_model.
    """
    df = fetch_stock_data(ticker)
    logging.debug(f"_train_and_register({model_name}, {ticker}) df_shape={df.shape}")
    X, y = _prepare_features_and_target(df)
    model_path = os.path.join(MODELS_DIR, f"{model_name}_{ticker}.pkl")

    if model_name == "rf":
        model = train_rf_model(X, y, model_path)
    elif model_name == "xgb":
        model = train_xgb_model(X, y, model_path)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    registry_manager.update_model(ticker, model_name, model_path, FEATURE_COLS)
    return model

def _train_and_register_global(model_name: str, ticker: str, retrain_days: int = 7) -> object:
    """
    Train or update a global RF/XGB model:
    - RF: retrain fully if any ticker is stale/missing
    - XGB: retrain fully if any ticker is stale/missing
    """
    # Load existing global model if there is one, because if there is one we want
    # the list of tickers in it si we can also train for those for the new iteration
    model_info = registry_manager.get_model_info("global", model_name)

    if model_info:
        # The tickers to train will be all past tickers and the one new ticker
        tickers_to_train = set(model_info.get("tickers_included", {}).keys())  # existing tickers
        tickers_to_train.add(ticker)  # add the new one
        # # Get the model path so we can override that model .pkl file
        # model_path = model_info["model_path"]
        # model = _load_model_from_path(model_path)
        # logging.debug(f"Loaded existing {model_name} global model from {model_path}")
    else:
        # If no model existed to begin this we just train the new ticker given
        tickers_to_train = {ticker}
        
    # Either way the model will be redone since load checks for staleness already. So we can set path and say model none
    model_path = os.path.join(MODELS_DIR, f"{model_name}_global.pkl")
    model = None
    # Also can get all global feature and target data already
    X_global, y_global = _stack_features_for_tickers(tickers_to_train)


    # -------------------- RF --------------------
    if model_name == "rf":
        model = train_rf_model(X_global, y_global, model_path)
        logging.debug(f"Trained RF global model on tickers: {tickers_to_train}")

    # -------------------- XGB --------------------
    elif model_name == "xgb":
        model = train_xgb_model(X_global, y_global, model_path)
        logging.debug(f"Trained XGB global model on tickers: {tickers_to_train}")

    else:
        raise ValueError(f"Unsupported global model: {model_name}")

    # just so we can  use same names for now
    tickers_included = tickers_included = {t: None for t in tickers_to_train}
    last_trained = format_timestamp(datetime.now(timezone.utc))
    # Update registry
    registry_manager.update_model("global", model_name, model_path, FEATURE_COLS, tickers_included, last_trained)
    return model


def _train_and_register_arima(ticker: str):
    """
    Train ARIMA(1,1,1) for a ticker robustly:
    - Handles short series by returning last observed value.
    - Interpolates missing values.
    - Uses safe start_params and limits optimizer iterations.
    """

    # Fetch and prepare data
    df = fetch_stock_data(ticker)
    print(f"[DEBUG] Training ARIMA for {ticker}, df_shape: {df.shape}")

    if df.index.freq is None:
        df = df.asfreq(pd.infer_freq(df.index) or "D")  # fallback daily

    close_series = df["Close"].astype(float).interpolate(method="linear").ffill().bfill()

    # Handle very short series
    if len(close_series) < 5:
        print(f"[WARN] Series too short for ARIMA, returning last observed value")
        class DummyARIMA:
            def forecast(self, steps=1):
                return pd.Series([close_series.iloc[-1]] * steps)
        return DummyARIMA()

    # Compute safe start_params
    phi1, theta1, intercept = _get_arima_start_params(close_series)
    phi1 = phi1 if phi1 is not None else 0.0
    theta1 = theta1 if theta1 is not None else 0.0
    intercept = intercept if intercept is not None else close_series.mean()
    print(f"[DEBUG] ARIMA start params phi1={phi1:.4f}, theta1={theta1:.4f}, intercept={intercept:.4f}")

    # Train ARIMA with warnings suppressed and limited iterations
    model = ARIMA(close_series, order=(1, 1, 1))
    model_fit = model.fit(start_params=[phi1, theta1, intercept], method_kwargs={"maxiter": 50})

    # Save model and update registry
    model_path = os.path.join(MODELS_DIR, f"arima_{ticker}.pkl")
    joblib.dump({"model": model_fit}, model_path)
    registry_manager.update_model(ticker, "arima", model_path, feature_cols=["Close"])
    print(f"[DEBUG] ARIMA model saved and registry updated for {ticker}")

    return model_fit



# -----------------------
# Load Model Function
# -----------------------
def load_model(model_name: str, ticker: str, retrain_days: int = 7, use_global: bool = True) -> object:
    """
    Load a model (per-ticker or global):
    - ARIMA: always per-ticker
    - RF/XGB: load global if use_global=True, otherwise per-ticker
    """
    model_name_lower = model_name.lower()

    # -------------------- ARIMA --------------------
    if model_name_lower == "arima":
        info = registry_manager.get_model_info(ticker, "arima")
        if not info or registry_manager.is_model_stale(ticker, "arima", retrain_days):
            return _train_and_register_arima(ticker)
        return _load_model_from_path(info["model_path"])

    # -------------------- Global model --------------------
    if use_global:
        info = registry_manager.get_model_info("global", model_name_lower)
        needs_retrain = not info or registry_manager.is_global_model_needs_update(model_name_lower, ticker, retrain_days)
        if needs_retrain:
            # training for the global model is only called if training is needed, so inside the global trainer I do not need to check staleness again
            return _train_and_register_global(model_name_lower, ticker, retrain_days)
        return _load_model_from_path(info["model_path"])


    # -------------------- Per-ticker RF/XGB --------------------
    info = registry_manager.get_model_info(ticker, model_name_lower)
    if not info or registry_manager.is_model_stale(ticker, model_name_lower, retrain_days):
        return _train_and_register(model_name_lower, ticker)

    return _load_model_from_path(info["model_path"])


def _train_and_register_global_batch(model_name: str, tickers: list, retrain_days: int = 7) -> object:
    """
    Train a global RF/XGB model on a list of tickers at once.
    Only used by the batch endpoint to save repeated training.
    """
    if not tickers:
        raise ValueError("Ticker list cannot be empty")

    # Stack features for all tickers
    X_global, y_global = _stack_features_for_tickers(tickers)

    # Determine model path
    model_path = os.path.join(MODELS_DIR, f"{model_name}_global.pkl")
    model = None

    # -------------------- RF --------------------
    if model_name.lower() == "rf":
        model = train_rf_model(X_global, y_global, model_path)
        logging.debug(f"Trained RF global batch model on tickers: {tickers}")

    # -------------------- XGB --------------------
    elif model_name.lower() == "xgb":
        model = train_xgb_model(X_global, y_global, model_path)
        logging.debug(f"Trained XGB global batch model on tickers: {tickers}")

    else:
        raise ValueError(f"Unsupported global model: {model_name}")

    # Update registry with last_trained and tickers_included
    last_trained = format_timestamp(datetime.now(timezone.utc))
    tickers_included = {t: last_trained for t in tickers}

    registry_manager.update_model(
        ticker="global",
        model_name=model_name.lower(),
        model_path=model_path,
        feature_cols=FEATURE_COLS,
        tickers_included=tickers_included,
        last_trained=last_trained
    )

    return model