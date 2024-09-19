import datetime
import pandas as pd
from models.train_models import train_model
from utils.data_utils import fetch_historical_market_data
from backtest import backtest  # Import the backtest function

def main():
    print("Starting the training process...")
    start = datetime.datetime.now() - datetime.timedelta(days=2)
    end_date = start.strftime('%Y-%m-%d')
    # Fetch data from five years ago today
    start_date = (start - datetime.timedelta(days=5*30)).strftime('%Y-%m-%d')
    print(f"Fetching data from {start_date} to {end_date}")
    
    # Fetch historical data
    data = fetch_historical_market_data('binance', 'BTC/USDT', timeframe='1m', start_date=start_date, end_date=end_date)
    
    # Train the model
    model = train_model(data)
    
    # Run backtest with $10,000
    if model is not None:
        print("Running backtest...")
        trade_log, final_value, profit = backtest(data, model, initial_balance=10000)
        
        # Output results
        for trade in trade_log:
            print(trade)
        print(f"Backtest complete. Final balance: ${final_value:.2f}, Profit: ${profit:.2f}")
    else:
        print("Model training failed, skipping backtest.")

if __name__ == "__main__":
    main()
