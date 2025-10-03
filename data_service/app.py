from fastapi import FastAPI
import yfinance as yf
from  fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Data Service")

#Gateway can call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"])

@app.get("/stock/{ticker}")
async def get_stock_data(ticker: str, period:str='5d', interval:str='1d'):
    """
    Fetch historical stock data for a given ticker symbol.

    Parameters:
    - ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc.).
    - period (str): The period over which to fetch data (default is '5d' for 5 days).
    - interval (str): The data interval (default is '1d' for daily data).

    Returns:
    - dict: A dictionary containing the recent OHLCV stock data.
    """
    data= yf.Ticker(ticker).history(period=period, interval=interval)
    # Conver to JSON-friendly format
    return data.reset_index().to_dict(orient='records')