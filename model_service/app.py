# model_service/app.py
import json
import os
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd

from .model_manager import load_model, _train_and_register, _train_and_register_global, _train_and_register_arima, _train_and_register_global_batch, registry_manager, FEATURE_COLS
from .model_manager import MODELS_DIR, REGISTRY_PATH

app = FastAPI(title="Model Service")

# ---------------------------
# Pydantic model for input
# ---------------------------
class FeaturesData(BaseModel):
    ticker: str
    features: List[Dict]  # list of feature dictionaries per date
    
class TrainBatchRequest(BaseModel):
    tickers: List[str]
    model_name: str
    use_global: bool = True  # only relevant for ML models (RF/XGB)


@app.get("/delete_models")
def delete_models():
    try:
        # Delete all model files in MODELS_DIR
        if os.path.exists(MODELS_DIR):
            for filename in os.listdir(MODELS_DIR):
                file_path = os.path.join(MODELS_DIR, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            # Optional: remove empty folders if any
        else:
            os.makedirs(MODELS_DIR, exist_ok=True)

        # Reset registry.json
        with open(REGISTRY_PATH, "w") as f:
            json.dump({}, f)

        return {"status": "success", "message": "All models deleted and registry cleared."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete models: {str(e)}")
    

# ---------------------------
# Prediction endpoint
# ---------------------------
@app.post("/predict/{model_name}")
def predict_model(model_name: str, data: FeaturesData, use_global: bool = Query(True, description="Use global model or not")):
    """
    Dynamic model prediction endpoint for ARIMA, RF, XGB, etc.
    - use_global: whether to use/update the global model for RF/XGB
    - Returns debug info about which tickers were retrained if applicable
    """
    print("Use global: ", use_global)
    try:
        print(f"[DEBUG] Incoming request for ticker: {data.ticker}, model: {model_name}, use_global={use_global}")

        # Convert incoming features to DataFrame
        df = pd.DataFrame(data.features)
        print(f"[DEBUG] Incoming features DataFrame shape: {df.shape}")
        print(f"[DEBUG] Incoming columns: {df.columns.tolist()}")

        if "Date" in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values('Date', inplace=True)
            print(f"[DEBUG] Sorted by Date")

        model_name_lower = model_name.lower()
        response_debug = {}

        # ---------------------------
        # ARIMA model
        # ---------------------------
        if model_name_lower == "arima":
            if "Close" not in df.columns:
                raise HTTPException(status_code=400, detail="ARIMA requires 'Close' column")
            model = load_model("arima", data.ticker, use_global=use_global)
            forecast = model.forecast(steps=1)
            next_day_price = forecast.iloc[0]
            print(f"[DEBUG] ARIMA forecasted value: {next_day_price}")

        # ---------------------------
        # ML models: RF, XGB
        # ---------------------------
        else:
            missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
            if missing_cols:
                print(f"[DEBUG] Missing feature columns: {missing_cols}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing feature columns for {model_name_lower}: {missing_cols}"
                )

            X = df[FEATURE_COLS].values
            print(f"[DEBUG] Feature matrix X shape: {X.shape}")

            if use_global:
                print(f"[DEBUG] Using global model for {model_name_lower}")
                model = _train_and_register_global(model_name_lower, data.ticker)
                global_info = registry_manager.get_model_info("global", model_name_lower)
                response_debug["tickers_in_global_model"] = list(global_info.get("tickers_included", {}).keys())
                response_debug["last_trained_per_ticker"] = global_info.get("tickers_included", {})
                print(f"[DEBUG] Global model tickers: {response_debug['tickers_in_global_model']}")
            else:
                print(f"[DEBUG] Using per-ticker model for {data.ticker}")
                model = load_model(model_name_lower, data.ticker, use_global=False)

            next_day_price = float(model.predict([X[-1]])[0])
            print(f"[DEBUG] Predicted next day price: {next_day_price}")

        return {
            "ticker": data.ticker,
            "model": model_name_lower,
            "prediction": round(float(next_day_price), 2),
            "used_global_model": use_global if model_name_lower != "arima" else None,
            "debug": response_debug
        }

    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------
# Batch prediction endpoint
# ---------------------------
@app.post("/predictbatch/{model_name}")
def predict_batch(model_name: str, batch_data: List[FeaturesData], use_global: bool = True):
    """
    Batch prediction endpoint for RF/XGB or ARIMA.
    """
    print("BATCH PREDICTOR")
    # Print the full incoming JSON payload
    print("Incoming batch JSON:", [item.model_dump() for item in batch_data])
    
    print("FEATURE_COLS:", FEATURE_COLS)


    response = []

    for entry in batch_data:
        ticker = entry.ticker
        df = pd.DataFrame(entry.features)  # convert list of dicts to DataFrame
        print(f"Ticker: {ticker}, Features DataFrame:")
        print(df)
        
        try:
            model_name_lower = model_name.lower()
            
            # -------------------- ARIMA --------------------
            if model_name_lower == "arima":
                model = load_model("arima", ticker, use_global=use_global)
                preds = model.forecast(steps=len(df)).tolist()
            else:
                print("here")
                # -------------------- RF/XGB --------------------
                X = df[FEATURE_COLS].values
                if use_global:
                    model = _train_and_register_global(model_name_lower, ticker)
                else:
                    model = load_model(model_name_lower, ticker, use_global=False)
                preds = model.predict(X).tolist()

            response.append({"ticker": ticker, "predictions": preds})

        except Exception as e:
            response.append({"ticker": ticker, "error": str(e)})

    print("Predit Batch Data: ", response)
    return response



# ---------------------------
# Manual train/retrain endpoint
# ---------------------------
@app.post("/train/{model_name}/{ticker}")
def train_model(model_name: str, ticker: str, use_global: bool = Query(True)):
    """
    Manually train or retrain a model for a given ticker.
    
    Supports:
    - ARIMA (per-ticker)
    - ML models (RF/XGB) with optional global model
    """
    model_name_lower = model_name.lower()
    
    try:
        if model_name_lower == "arima":
            # ARIMA is always per-ticker
            model = _train_and_register_arima(ticker)
            info = registry_manager.get_model_info(ticker, "arima")
            used_global = None
        else:
            # RF/XGB: handle global vs per-ticker
            if use_global:
                model = _train_and_register_global(model_name_lower, ticker)
                info = registry_manager.get_model_info("global", model_name_lower)
            else:
                model = _train_and_register(model_name_lower, ticker)
                info = registry_manager.get_model_info(ticker, model_name_lower)
            used_global = use_global
        
        return {
            "ticker": ticker,
            "model": model_name_lower,
            "status": "trained",
            "trained_at": info["last_trained_at"],
            "used_global_model": used_global
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


# ---------------------------
# Batch train/retrain endpoint
# ---------------------------
@app.post("/trainbatch")
def train_batch(req: TrainBatchRequest):
    """
    Train multiple tickers at once:
    - If use_global=True and model is RF/XGB: train global model once for all tickers
    - If use_global=False:
        - ARIMA: train per-ticker
        - RF/XGB: train per-ticker
    """
    model_name_lower = req.model_name.lower()

    if not req.tickers:
        raise HTTPException(status_code=400, detail="Ticker list cannot be empty.")

    trained_tickers = []
    response_debug = {}

    try:
        # ----------------------------
        # Global batch training (RF/XGB)
        # ----------------------------
        if use_global := req.use_global and model_name_lower in ["rf", "xgb"]:
            model = _train_and_register_global_batch(model_name_lower, req.tickers)
            info = registry_manager.get_model_info("global", model_name_lower)
            return {
                "model": model_name_lower,
                "tickers_trained": req.tickers,
                "status": "trained",
                "trained_at": info["last_trained_at"],
                "tickers_in_global_model": list(info.get("tickers_included", {}).keys())
            }

        # ----------------------------
        # Per-ticker training
        # ----------------------------
        for ticker in req.tickers:
            if model_name_lower == "arima":
                _train_and_register_arima(ticker)
            elif model_name_lower in ["rf", "xgb"]:
                _train_and_register(model_name_lower, ticker)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported model: {model_name_lower}")

            trained_tickers.append(ticker)

        # Gather registry info for first ticker (last_trained will be roughly same for all)
        if model_name_lower == "arima":
            last_trained = registry_manager.get_model_info(trained_tickers[0], "arima")["last_trained_at"]
        else:
            last_trained = registry_manager.get_model_info(trained_tickers[0], model_name_lower)["last_trained_at"]

        return {
            "model": model_name_lower,
            "tickers_trained": trained_tickers,
            "status": "trained",
            "trained_at": last_trained,
            "used_global_model": False
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch training failed: {e}")
