import pandas as pd
import joblib

def backtest(data, model, initial_balance=10000):
    print("Starting backtest...")

    # Create features again for the test data
    from utils.features_engineering import create_features
    data = create_features(data)

    # Ensure required columns exist
    required_columns = [
        'price_change', 'percent_change', 'log_return', 
        'short_mavg', 'long_mavg', 'momentum', 'volatility', 
        'macd', 'macdsignal', 'macdhist', 'rsi', 'hour', 'day_of_week'
    ]
    
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {', '.join(required_columns)}")
    
    # Drop NaN values
    data.dropna(inplace=True)

    # Prepare features and target
    X = data[required_columns]
    
    # Get model predictions
    y_pred = model.predict(X)
    
    # Initialize backtesting variables
    balance = initial_balance
    btc_position = 0
    trade_log = []
    
    # Simulate trading
    for i in range(len(y_pred)):
        signal = y_pred[i]  # 1 = Buy, 0 = Sell
        current_price = data['close'].iloc[i]
        
        # Buy Signal (1)
        if signal == 1 and btc_position == 0:  # Only buy if no current BTC position
            btc_position = balance / current_price  # Buy BTC with all available balance
            balance = 0
        
        # Sell Signal (0)
        elif signal == 0 and btc_position > 0:  # Only sell if holding BTC
            balance = btc_position * current_price  # Sell all BTC
            
            btc_position = 0
    
    # Final balance (account for any remaining BTC holdings)
    final_value = balance + (btc_position * data['close'].iloc[-1])  # Final portfolio value
    profit = final_value - initial_balance
    
    print(f"Initial Balance: ${initial_balance}")
    print(f"Final Balance: ${final_value:.2f}")
    print(f"Total Profit: ${profit:.2f}")
    
    # Return trade log and final results
    return trade_log, final_value, profit
