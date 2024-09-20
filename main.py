import datetime
import pandas as pd
from models.train_models import train_xgboost_model_with_random_search
from utils.data_utils import fetch_historical_market_data
from backtest import backtest  # Assuming you have the backtest function
from utils.load_data import load_data

def main():
    print("Starting the training process...")

    data = load_data("data.csv")

    if data.empty:
        # Define the date range (last 5 years)
        start = datetime.datetime.now() - datetime.timedelta(days=2)
        end_date = start.strftime('%Y-%m-%d')
        start_date = (start - datetime.timedelta(days=365*5)).strftime('%Y-%m-%d')

        print(f"Fetching data from {start_date} to {end_date}")

        # Fetch historical market data for backtesting
        data = fetch_historical_market_data('binance', 'BTC/USDT', timeframe='1m', start_date=start_date, end_date=end_date)
    
    # Train the XGBoost model with GridSearchCV
    model = train_xgboost_model_with_random_search(data)

    if model is not None:
        print("Model trained successfully. Now running backtest...")
        backtest(data, model, initial_balance=10000)
    else:
        print("Model training failed.")

if __name__ == "__main__":
    main()
