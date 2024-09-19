from utils.preprocessing import create_features, preprocess_data, create_sequences
import numpy as np
import pandas as pd
from tensorflow import keras

def backtest_adam():
    # Load the trained LSTM model
    model = keras.models.load_model('adam.h5')
    
    # Load the raw data
    data = pd.read_csv('data.csv')
    
    # Step 1: Feature engineering (create features)
    data = create_features(data)  # Ensure this creates the necessary features

    # Step 2: Preprocess the data (scaling and encoding)
    data = preprocess_data(data)  # Assuming you have this function defined

    # Initialize backtesting variables
    btc_position = 0
    initial_balance = 10000  # Starting balance in USD
    balance = initial_balance
    trade_log = []

    # Define sequence length used during training
    seq_length = 10

    # Prepare sequences for predictions
    X = []  # List to store sequences of past data
    for i in range(seq_length, len(data)):
        X.append(data.iloc[i-seq_length:i].values)
    
    X = np.array(X)
    
    # Run the backtest over the data
    for i in range(seq_length, len(data)):
        current_price = data.iloc[i]['close']  # Assuming 'close' is the target column
        sequence = X[i-seq_length]  # Get the ith sequence of data points for prediction
        sequence = np.expand_dims(sequence, axis=0)  # Reshape for model prediction
        
        # Make prediction (e.g., 1 for Buy, 0 for Sell)
        prediction = model.predict(sequence)[0][0]
        
        # Trading strategy: Buy if prediction > 0.5, Sell if prediction <= 0.5
        if prediction > 0.5 and btc_position == 0:  # Buy signal
            btc_position = balance / current_price  # Buy BTC with all available balance
            balance = 0  # No more USD after buying
            trade_log.append(f"Bought BTC at {current_price} with position size {btc_position:.4f}")
        
        elif prediction <= 0.5 and btc_position > 0:  # Sell signal
            balance = btc_position * current_price  # Sell all BTC
            btc_position = 0  # No more BTC after selling
            trade_log.append(f"Sold BTC at {current_price}, balance: {balance:.2f} USD")

    # Final portfolio value
    final_value = balance if btc_position == 0 else btc_position * data.iloc[-1]['close']
    profit = final_value - initial_balance

    return trade_log, final_value, profit

if __name__ == "__main__":
    trade_log, final_value, profit = backtest_adam()
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Total Profit: ${profit:.2f}")
