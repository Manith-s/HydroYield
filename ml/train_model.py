"""
Train the Random Forest model for crop prediction
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Load and preprocess the crop dataset
    
    Args:
        file_path (str): Path to the CSV data file
        
    Returns:
        tuple: X (features), y (labels)
    """
    try:
        # Load data
        logger.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        
        # Basic info
        logger.info(f"Dataset shape: {data.shape}")
        logger.info(f"Columns: {data.columns.tolist()}")
        
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Missing values found: {missing_values[missing_values > 0]}")
            # Fill missing values with median
            data = data.fillna(data.median())
        
        # Split features and target
        X = data.drop('label', axis=1)
        y = data['label']
        
        # Log class distribution
        class_dist = y.value_counts()
        logger.info(f"Class distribution: {class_dist}")
        
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def train_model(X, y, model_path):
    """
    Train and save the Random Forest model
    
    Args:
        X (DataFrame): Features
        y (Series): Target labels
        model_path (str): Path to save the model
        
    Returns:
        tuple: Model and accuracy score
    """
    try:
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize model
        rf = RandomForestClassifier(random_state=42)
        
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Perform grid search with cross-validation
        logger.info("Performing grid search for hyperparameter tuning")
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate model
        y_pred = best_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy: {accuracy:.4f}")
        
        # Print classification report
        report = classification_report(y_test, y_pred)
        logger.info(f"Classification report:\n{report}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Feature importance:\n{feature_importance}")
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Also save the scaler
        scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        return best_model, accuracy
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

if __name__ == "__main__":
    try:
        # Check command-line arguments
        if len(sys.argv) > 1:
            data_path = sys.argv[1]
        else:
            # Default path
            data_path = os.path.join("data", "cpdata.csv")
        
        # Set model output path
        model_output = os.path.join("ml", "models", "random_forest_model.pkl")
        
        # Load data
        X, y = load_data(data_path)
        
        # Train model
        model, accuracy = train_model(X, y, model_output)
        
        logger.info(f"Training completed successfully with accuracy: {accuracy:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)