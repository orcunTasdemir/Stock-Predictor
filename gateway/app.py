from fastapi import FastAPI, HTTPException, Query
import httpx


app = FastAPI(title="API Gateway") 

#DATA_SERVICE_URL = "http://data_service:8001" #Docker service name
DATA_SERVICE_URL = "http://127.0.0.1:8001" #Docker service name
FEATURE_SERVICE_URL = "http://127.0.0.1:8002"
MODEL_SERVICE_URL = "http://127.0.0.1:8003"


# UNCOMMENT IF YOU WANT TO EXPOSE RAW DATA TO THE CLIENT SIDE

# @app.get("/predict/stock/{ticker}")
# def get_stock_from_data_service(ticker: str):
#     """
#     Calls the data service and returns stock data to the client
#     """
#     try:
#         response = httpx.get(f"{DATA_SERVICE_URL}/stock/{ticker}")
#         response.raise_for_status()
#     except httpx.RequestError as e:
#         raise HTTPException(status_code=500, detail=f"Data service is unreachable: {e}")
#     except httpx.HTTPStatusError as e:
#         raise HTTPException(status_code=response.status_code, detail=str(e))
#     return response.json()


# UNCOMMENT IF YOU WANT TO EXPOSE FEATURES TO THE CLIENT SIDE

# @app.get("/predict/features/{ticker}")
# def get_features(ticker: str):
#     """
#     Calls Data Service -> Feature Service -> Returns enriched features to the client
#     """
#     try:
#         # Get raw stock data
#         data_response = httpx.get(f"{DATA_SERVICE_URL}/stock/{ticker}")
#         data_response.raise_for_status()
#         raw_data = data_response.json()
#     except httpx.RequestError as e:
#         raise HTTPException(status_code=500, detail=f"Data service is unreachable: {e}")
#     except httpx.HTTPStatusError as e:
#         raise HTTPException(status_code=data_response.status_code, detail=str(e))
    
#     # Send raw data to the feature service
#     try:
#         feature_response = httpx.post(f"{FEATURE_SERVICE_URL}/features", json={"ticker": ticker, "raw_data": raw_data})
#         feature_response.raise_for_status()
#         features = feature_response.json()
#     except httpx.RequestError as e:
#         raise HTTPException(status_code=500, detail=f"Feature service is unreachable: {e}")
#     except httpx.HTTPStatusError as e:
#         raise HTTPException(status_code=feature_response.status_code, detail=str(e))
#     # Return enriched features to the client
#     return features

@app.get("/predict/{ticker}")
def predict_stock(ticker: str, model: str = Query("arima")):
    """
    Full pipeline: Data Service -> Feature Service -> Model Service -> Returns prediction to the client
    Supports 'arima' and 'rf' models right now
    """
    try:
        # Get raw stock data
        data_response = httpx.get(f"{DATA_SERVICE_URL}/stock/{ticker}")
        data_response.raise_for_status()
        raw_data = data_response.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Data service is unreachable: {e}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=data_response.status_code, detail=str(e))
    
    # Send raw data to the feature service
    try:
        feature_response = httpx.post(f"{FEATURE_SERVICE_URL}/features", json={"ticker": ticker, "raw_data": raw_data})
        feature_response.raise_for_status()
        features = feature_response.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Feature service is unreachable: {e}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=feature_response.status_code, detail=str(e))
    
    # Send features to the model service for prediction
    try:
        model_response = httpx.post(f"{MODEL_SERVICE_URL}/predict/{model.lower()}", json={"ticker": ticker, "features": features['features']})
        model_response.raise_for_status()
        prediction = model_response.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Model service is unreachable: {e}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=model_response.status_code, detail=str(e))
    
    # Return prediction to the client
    return prediction