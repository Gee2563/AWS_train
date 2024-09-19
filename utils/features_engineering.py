import pandas as pd
import numpy as np

def create_features(data):
    # Price-based features
    data['price_change'] = data['close'].diff(1)
    data['percent_change'] = data['close'].pct_change(1)
    data['log_return'] = np.log(data['close'] / data['close'].shift(1))

    # Momentum features
    data['short_mavg'] = data['close'].rolling(window=5).mean()
    data['long_mavg'] = data['close'].rolling(window=30).mean()
    data['momentum'] = data['close'] - data['close'].shift(10)
    data['volatility'] = data['close'].rolling(window=10).std()

    # MACD features
    ema_12 = data['close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = ema_12 - ema_26
    data['macdsignal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macdhist'] = data['macd'] - data['macdsignal']

    # Relative Strength Index (RSI)
    delta = data['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # Time-based features
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek

    # Drop NaN values
    data.dropna(inplace=True)
    
    return data
