import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
from utils.features_engineering import create_features

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(data):
    try:
        logger.info("Preparing data...")

        # Create features based on short MAVG, long MAVG, RSI, and MACD
        print('Features')
        data = create_features(data)
        
        if data.empty:
            logger.warning("No data available after feature creation.")
            return None
        print('Features created')

        logger.info("Feature creation completed.")

        # Ensure required columns exist
        required_columns = ['short_mavg', 'long_mavg', 'rsi', 'macd']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {', '.join(required_columns)}")

        # Remove NaN values
        data.dropna(inplace=True)

        # Prepare features and target
        X = data[required_columns]
        y = (data['close'].shift(-1) > data['close']).astype(int)

        # Remove rows with NaN values resulting from shift
        X = X[:-1]
        y = y[:-1]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logger.info("Training the model...")
        # Initialize and train the model
        model = RandomForestClassifier(
            n_estimators=100,  # Adjust the number of trees as needed
            max_depth=10,     # Limit the maximum depth of the trees
            n_jobs=-1,        # Use all available cores
            random_state=42
        )
        model.fit(X_train, y_train)

        logger.info("Model training completed.")

        # Make predictions and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"Model Accuracy: {accuracy:.2f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        # Save the trained model
        joblib.dump(model, 'random_forest_model.pkl')
        logger.info("Model saved as random_forest_model.pkl")
        # Once stored - run scp random_forest_model.pkl ubuntu@34.123.45.67:/home/ubuntu/models/ in CLI


        return model

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
    except Exception as e:
        logger.error(f"Error: {e}")
    
    return None
