# utils/data_utils.py
import ccxt
import pandas as pd


# this function fetches historical market data from an exchange - research suggests 5 years worth of data = 131MB - will store in CSV 

def fetch_historical_market_data(exchange_name, symbol, timeframe='1m', start_date=None, end_date=None, limit=1000):
    print(f"Fetching historical data for {exchange_name} {symbol} {timeframe}")
    exchange = getattr(ccxt, exchange_name)()
    if start_date is not None:
        # make the start date is in ms
        start_timestamp = int(pd.Timestamp(start_date).timestamp() * 1000) 
    else:
        start_timestamp = None
    if end_date is not None:
        # make the end date is in ms
        end_timestamp = int(pd.Timestamp(end_date).timestamp() * 1000)
    else:
        end_timestamp = None
    all_data = []

    while True:
        # fetch open, high, low, close, volume data from ccxt
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=start_timestamp)
        if not ohlcv:
            break
        # convert the data into a pandas dataframe and add to data-list
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        all_data.append(data)
        # update the start timestamp for the next fetch
        start_timestamp  = ohlcv[-1][0] + 1
        # when the end timestamp is reached, break the loop
        if end_timestamp and start_timestamp >= end_timestamp:
            break

        # add the fetched data to the historical data
    historical_data = pd.concat(all_data, ignore_index=True)
    # convert the timestamp to datetime
    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'], unit='ms')
    # set the timestamp as the index
    historical_data.set_index('timestamp', inplace=True)

    return historical_data

