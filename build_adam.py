import pandas as pd
import numpy as np
from utils.preprocessing import create_features, preprocess_data, create_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import joblib
from backtest import backtest

# Define the function to build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Load your data
    print("Starting the training process...")
    data = pd.read_csv('data.csv', parse_dates=True, index_col='timestamp')

    
    # Step 1: Feature creation and preprocessing
    data = create_features(data)  # Feature engineering
    data = preprocess_data(data)  # Scaling and encoding categorical features

    # Step 2: Add a target column (e.g., predict next-minute price direction)
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)  # Binary classification

    # Drop rows with NaN values (after creating target column)
    data.dropna(inplace=True)

    # Step 3: Create sequences for LSTM/GRU (time-series)
    seq_length = 10  # Number of previous time steps
    X, y = create_sequences(data, seq_length=seq_length)

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Step 4: Split into training and test sets
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Convert data types to float32 and int32
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)

    # Step 5: Build and train the LSTM model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)

    # Step 6: Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    print("Model training completed.")
    # Step 7: Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.2f}")
    # Step 8: Save the model
    model.save('adam.h5')
    

if __name__ == "__main__":
    main()
