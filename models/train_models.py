import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
from utils.features_engineering import create_features
import matplotlib.pyplot as plt
from xgboost import plot_importance

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_xgboost_model_with_grid_search(data):
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

        # Remove rows with NaN values resulting from shift
        X = X[:-1]
        y = y[:-1]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [50, 100, 200,300],  # Number of trees
            'max_depth': [5, 10, 15,20,50],        # Maximum depth of the trees
            'learning_rate': [0.01, 0.05, 0.1],  # Learning rate for boosting
        }

        # Initialize the XGBoost model
        xgb_model = XGBClassifier(eval_metric='logloss', n_jobs=-1, random_state=42)

        # Perform grid search with cross-validation
        logger.info("Starting GridSearchCV for XGBoost...")
        grid_search = GridSearchCV(xgb_model, param_grid, cv=4, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train)

        # Get the best parameters from the grid search
        best_params = grid_search.best_params_
        logger.info(f"Best parameters from Grid Search: {best_params}")

        # Retrain the model using the best parameters
        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)

        logger.info("Model training with best parameters completed.")

        # Make predictions and evaluate
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model Accuracy: {accuracy:.2f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        # Feature Importance
        plot_importance(best_model, importance_type='weight')
        plt.title("Feature Importance - XGBoost")
        plt.show()

        # Save the trained model
        joblib.dump(best_model, 'xgboost_model_best.pkl')
        logger.info("Best model saved as xgboost_model_best.pkl")

        return best_model

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
    except Exception as e:
        logger.error(f"Error: {e}")
    
    return None
