# model_service/registry_manager.py

import json
import os
from datetime import datetime, timezone
from .utils import format_timestamp, is_stale

class RegistryManager:
    """
    Handles loading,saving, and updating the model registry.
    Always treats the JSON file as the source of truth.
    """
    
    def __init__(self, registry_path:str):
        self.registry_path = registry_path
        self._cache = None # optional in memory cache
        self._cache_time = None
        
        # ensure the registry.json exists
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, "w") as f:
                json.dump({}, f, indent=2)
                
    def load(self, force:bool = False) -> dict:
        """Load the registry from disk. Force reload bypass cache."""
        if force or self._cache is None:
            with open(self.registry_path, "r") as f:
                try:
                    self._cache = json.load(f)
                except json.JSONDecodeError:
                    self._cache = {} # Empty the register if the file is corrupted
            self._cache_time = datetime.now(timezone.utc)
        return self._cache
    
    def save(self):
        """Save the current cache to disk."""
        if self._cache is not None:
            with open(self.registry_path, "w") as f:
                json.dump(self._cache, f, indent=2) 
                
                
    def update_model(
        self,
        ticker: str,
        model_name: str,
        model_path: str,
        feature_cols: list = None,
        tickers_included: dict = None,
        last_trained: str = None,
    ):
        """
        Update a model entry in the registry and persist immediately.

        Args:
            ticker: stock ticker symbol or "global" for global models
            model_name: model type, e.g., "rf", "xgb", "arima"
            model_path: path to the saved model file
            feature_cols: list of feature column names used in training
            tickers_included: dict of tickers included (only for global models)
            last_trained: UTC timestamp string for the model (if None, use current time)
        """
        reg = self.load(force=True)
        if ticker not in reg:
            reg[ticker] = {}

        # Base entry
        entry = {
            "model_path": model_path,
            "last_trained_at": last_trained or format_timestamp(datetime.now(timezone.utc)),
            "feature_cols": feature_cols or [],
        }

        # Only add tickers_included for global models
        if ticker == "global":
            entry["tickers_included"] = tickers_included or {}

        reg[ticker][model_name.lower()] = entry
        self._cache = reg
        self.save()


        
    def get_model_info(self, ticker:str, model_name:str):
        """Get model info from registry, or None if not found."""
        reg = self.load(force=True)
        print("get model info return: ", reg.get(ticker, {}).get(model_name.lower()))
        return reg.get(ticker, {}).get(model_name.lower())
    
    def is_model_stale(self, ticker:str, model_name:str, retain_days:int) -> bool:
        """Check if a model is stale based on last_trained_at."""
        model_info = self.get_model_info(ticker, model_name)
        if not model_info or "last_trained_at" not in model_info:
            return True
        return is_stale(model_info["last_trained_at"], retain_days)
    
    def is_global_model_needs_update(self, model_name: str, ticker: str, retain_days: int) -> bool:
        """
        Determine if a global model needs updating for a given ticker.
        Returns True if:
        - global model does not exist, OR
        - ticker not in tickers_included, OR
        - any ticker in tickers_included is stale
        """
        info = self.get_model_info("global", model_name)
        #if model does not exist retrain. load already checks this but for now I will keep to not break anything
        if not info:
            return True

        #tickers included should be a list of tickers
        tickers_included = info.get("tickers_included", {})

        # Check if requested ticker is missing, if so return true
        if ticker not in tickers_included:
            return True

        # Check if last_trained_at is stale
        last_trained = info.get("last_trained_at")
        if not last_trained or is_stale(last_trained, retain_days):
            return True

        return False

        
            