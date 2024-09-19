import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
from utils.features_engineering import create_features
import matplotlib.pyplot as plt

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(data):
    try:
        logger.info("Preparing data...")

        # Create features
        data = create_features(data)
        if data.empty:
            logger.warning("No data available after feature creation.")
            return None
        logger.info("Feature creation completed.")
        
        # Include all relevant columns
        required_columns = [
            'price_change', 'percent_change', 'log_return',  # Price-based features
            'short_mavg', 'long_mavg', 'momentum', 'volatility',  # Momentum features
            'macd', 'macdsignal', 'macdhist',  # MACD features
            'rsi',  # RSI feature
            'hour', 'day_of_week'  # Time-based features
        ]

        # Check if required columns exist
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {', '.join(required_columns)}")
        
        # Remove NaN values
        data.dropna(inplace=True)

        # Prepare features and target
        X = data[required_columns]
        y = (data['close'].shift(-1) > data['close']).astype(int)

        # Remove rows with NaN values
        X = X[:-1]
        y = y[:-1]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the model with cross-validation
        logger.info("Training the model...")
        model = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
        
        model.fit(X_train, y_train)
        logger.info("Model training completed.")

        # Make predictions and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model Accuracy: {accuracy:.2f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        # Feature Importance
        feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
        feature_importances.sort_values().plot(kind='barh')
        plt.title("Feature Importances")
        plt.show()

        # Save the trained model
        joblib.dump(model, 'random_forest_model.pkl')
        logger.info("Model saved as random_forest_model.pkl")

        return model

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
    except Exception as e:
        logger.error(f"Error: {e}")
    
    return None
