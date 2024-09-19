import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

# Preprocessing function
def preprocess_data(data):
    # Define features for scaling
    feature_columns = [
        'price_change', 'percent_change', 'log_return', 'short_mavg', 'long_mavg',
        'momentum', 'volatility', 'macd', 'macdsignal', 'macdhist', 'rsi'
    ]
    
    # Scale features using MinMaxScaler
    scaler = MinMaxScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    
    # One-hot encode time-based categorical features
    data = pd.get_dummies(data, columns=['hour', 'day_of_week'], drop_first=True)
    
    return data

# Apply the full feature creation and preprocessing pipeline
data = pd.read_csv('data.csv', parse_dates=True, index_col='timestamp')
data = create_features(data)
data = preprocess_data(data)

# Now your data is ready for training a deep learning model

def create_sequences(data, seq_length=10):
    """
    Create sequences for LSTM/GRU models.
    
    Parameters:
    - data: The preprocessed data (as a pandas DataFrame or NumPy array).
    - seq_length: The number of time steps to look back for sequence creation.
    
    Returns:
    - X: The input sequences.
    - y: The target values (for the next time step).
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i + seq_length].values)
        y.append(data.iloc[i + seq_length]['target'])  # Assuming target column exists
    return np.array(X), np.array(y)