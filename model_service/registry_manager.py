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
                
    def update_model(self, ticker:str, model_name:str, model_path:str, feature_cols:list = []):
        "Update a model entry in the registry and persist immediately."
        reg = self.load(force=True)
        if ticker not in reg:
            reg[ticker] = {}
        reg[ticker][model_name.lower()] = {
            "model_path": model_path,
            "last_trained_at": format_timestamp(datetime.now(timezone.utc)),
            "feature_cols": feature_cols
        }
        self.save()
        
    def get_model_info(self, ticker:str, model_name:str):
        """Get model info from registry, or None if not found."""
        reg = self.load(force=True)
        return reg.get(ticker, {}).get(model_name.lower())
    
    def is_model_stale(self, ticker:str, model_name:str, retain_days:int) -> bool:
        """Check if a model is stale based on last_trained_at."""
        model_info = self.get_model_info(ticker, model_name)
        if not model_info or "last_trained_at" not in model_info:
            return True
        return is_stale(model_info["last_trained_at"], retain_days)
    
        