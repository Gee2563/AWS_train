import datetime
import pandas as pd
from models.train_models import train_model
from utils.data_utils import fetch_historical_market_data

def main():
    print("Starting the training process...")
    start = datetime.datetime.now() - datetime.timedelta(days=2)
    end_date = start.strftime('%Y-%m-%d')
    #end date = five years ago today
    start_date = (start - datetime.timedelta(days=5*30)).strftime('%Y-%m-%d')
    print(f"Fetching data from {start_date} to {end_date}")
    data = fetch_historical_market_data('binance', 'BTC/USDT', timeframe='1m', start_date=start_date, end_date=end_date)
    train_model(data)
    
   
if __name__ == "__main__":
    main()
