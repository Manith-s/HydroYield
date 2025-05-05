"""
Crop Yield Prediction using Random Forest
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("yield_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def prepare_yield_dataset(input_data, output_path=None):
    """
    Prepare dataset for yield prediction by adding yield data
    
    Args:
        input_data (str or DataFrame): Input data path or DataFrame
        output_path (str, optional): Path to save the prepared dataset
        
    Returns:
        DataFrame: Prepared dataset with yield data
    """
    try:
        # Load data if path is provided
        if isinstance(input_data, str):
            logger.info(f"Loading data from {input_data}")
            data = pd.read_csv(input_data)
        else:
            data = input_data.copy()
        
        # Check if data already has a yield column
        if 'yield' in data.columns:
            logger.info("Dataset already contains yield data")
            yield_data = data
        else:
            logger.info("Adding synthetic yield data based on optimal conditions")
            
            # Create yield column based on how close environmental factors are to optimal conditions
            # This is a simplified model - in a real scenario, you would use actual historical yield data
            yield_data = data.copy()
            
            # Define optimal ranges for each crop
            crop_optimal = {
                'rice': {'temperature': (20, 27), 'humidity': (80, 90), 'ph': (6.0, 7.5)},
                'wheat': {'temperature': (15, 24), 'humidity': (60, 70), 'ph': (5.5, 7.0)},
                'maize': {'temperature': (21, 32), 'humidity': (65, 85), 'ph': (5.8, 7.0)},
                # Add more crops as needed
            }
            
            # Calculate base yield for each crop (different crops have different baseline yields)
            base_yields = {
                'rice': 5000,  # kg/hectare
                'wheat': 3500, 
                'maize': 6000,
                # Add more crops with their base yields
            }
            
            # Calculate yield based on environmental factors
            yield_values = []
            
            for _, row in yield_data.iterrows():
                crop = row['label']
                base_yield = base_yields.get(crop, 4000)  # Default base yield
                
                # If we have optimal data for this crop
                if crop in crop_optimal:
                    optimal = crop_optimal[crop]
                    
                    # Calculate factor scores (0-1) based on how close to optimal
                    temp_score = calculate_factor_score(row['temperature'], optimal['temperature'])
                    humidity_score = calculate_factor_score(row['humidity'], optimal['humidity'])
                    ph_score = calculate_factor_score(row['ph'], optimal['ph'])
                    rainfall_factor = min(1.0, row['rainfall'] / 200)  # Simplified rainfall factor
                    
                    # Calculate yield adjustment factor (0.5 to 1.5)
                    adjustment = 0.5 + (temp_score * 0.3 + humidity_score * 0.3 + ph_score * 0.2 + rainfall_factor * 0.2)
                    
                    # Apply adjustment to base yield with some randomness
                    yield_value = base_yield * adjustment * (0.9 + 0.2 * np.random.random())
                else:
                    # For unknown crops, use a generic calculation
                    yield_value = base_yield * (0.7 + 0.6 * np.random.random())
                
                yield_values.append(yield_value)
            
            # Add yield column to the dataset
            yield_data['yield'] = yield_values
        
        # Save the prepared dataset if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            yield_data.to_csv(output_path, index=False)
            logger.info(f"Prepared dataset saved to {output_path}")
        
        return yield_data
    
    except Exception as e:
        logger.error(f"Error preparing yield dataset: {e}")
        raise

def calculate_factor_score(value, optimal_range):
    """
    Calculate a score (0-1) based on how close a value is to the optimal range
    
    Args:
        value (float): The environmental value
        optimal_range (tuple): The optimal range (min, max)
        
    Returns:
        float: Score between 0 and 1
    """
    min_val, max_val = optimal_range
    
    if min_val <= value <= max_val:
        # Value is in optimal range
        return 1.0
    
    # Calculate how far outside the range
    if value < min_val:
        distance = min_val - value
        ref_value = min_val
    else:  # value > max_val
        distance = value - max_val
        ref_value = max_val
    
    # Calculate score that decreases with distance
    score = max(0, 1 - (distance / (ref_value * 0.5)))
    return score

def train_yield_model(data, model_path):
    """
    Train a Random Forest model for yield prediction
    
    Args:
        data (DataFrame): Prepared dataset with yield data
        model_path (str): Path to save the trained model
        
    Returns:
        tuple: Trained model and evaluation metrics
    """
    try:
        logger.info("Training yield prediction model")
        
        # Prepare features and target
        X = data[['temperature', 'humidity', 'ph', 'rainfall']]
        y = data['yield']
        
        # One-hot encode the crop type
        if 'label' in data.columns:
            crop_dummies = pd.get_dummies(data['label'], prefix='crop')
            X = pd.concat([X, crop_dummies], axis=1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize model
        rf = RandomForestRegressor(random_state=42)
        
        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Train the model
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        
        # Evaluate the model
        y_pred = best_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate percentage error
        percentage_error = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Log evaluation metrics
        logger.info(f"Model evaluation:")
        logger.info(f"MSE: {mse:.2f}")
        logger.info(f"RMSE: {rmse:.2f}")
        logger.info(f"MAE: {mae:.2f}")
        logger.info(f"R² Score: {r2:.4f}")
        logger.info(f"Mean Percentage Error: {percentage_error:.2f}%")
        logger.info(f"Prediction Accuracy: {100 - percentage_error:.2f}%")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Feature importance:\n{feature_importance.head(10)}")
        
        # Save the model and scaler
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(best_model, model_path)
        
        scaler_path = os.path.join(os.path.dirname(model_path), 'yield_scaler.pkl')
        joblib.dump(scaler, scaler_path)
        
        # Also save feature names for later use
        feature_names_path = os.path.join(os.path.dirname(model_path), 'yield_feature_names.pkl')
        joblib.dump(list(X.columns), feature_names_path)
        
        logger.info(f"Model saved to {model_path}")
        
        # Return model and metrics
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'percentage_error': percentage_error,
            'accuracy': 100 - percentage_error,
            'best_params': grid_search.best_params_
        }
        
        return best_model, metrics
    
    except Exception as e:
        logger.error(f"Error training yield model: {e}")
        raise

def predict_yield(model_path, input_data):
    """
    Predict crop yield based on environmental parameters
    
    Args:
        model_path (str): Path to the trained model
        input_data (dict): Environmental data with crop type
        
    Returns:
        dict: Predicted yield and confidence
    """
    try:
        # Load model and related files
        model = joblib.load(model_path)
        
        scaler_path = os.path.join(os.path.dirname(model_path), 'yield_scaler.pkl')
        scaler = joblib.load(scaler_path)
        
        feature_names_path = os.path.join(os.path.dirname(model_path), 'yield_feature_names.pkl')
        feature_names = joblib.load(feature_names_path)
        
        # Prepare input data
        input_dict = {
            'temperature': input_data.get('temperature', 0),
            'humidity': input_data.get('humidity', 0),
            'ph': input_data.get('ph', 0),
            'rainfall': input_data.get('rainfall', 0)
        }
        
        # Create DataFrame with all features set to 0
        input_df = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Set environmental parameters
        for key, value in input_dict.items():
            if key in input_df.columns:
                input_df[key] = value
        
        # Set crop indicator if provided
        crop = input_data.get('crop')
        if crop:
            crop_column = f'crop_{crop}'
            if crop_column in input_df.columns:
                input_df[crop_column] = 1
        
        # Scale input data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        predicted_yield = model.predict(input_scaled)[0]
        
        # Get prediction interval using trees in the forest
        tree_predictions = [tree.predict(input_scaled)[0] for tree in model.estimators_]
        lower_bound = np.percentile(tree_predictions, 10)
        upper_bound = np.percentile(tree_predictions, 90)
        std_dev = np.std(tree_predictions)
        
        # Calculate confidence level based on prediction interval width
        interval_width = upper_bound - lower_bound
        mean_yield = np.mean(tree_predictions)
        confidence = max(0, min(100, 100 * (1 - (interval_width / (mean_yield * 2)))))
        
        # Return prediction results
        result = {
            'predicted_yield': round(predicted_yield, 2),
            'confidence': round(confidence, 2),
            'lower_bound': round(lower_bound, 2),
            'upper_bound': round(upper_bound, 2),
            'unit': 'kg/hectare',
            'factors': input_dict
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error predicting yield: {e}")
        
        # Fallback to a rule-based estimation
        return estimate_yield_rule_based(input_data)

def estimate_yield_rule_based(input_data):
    """
    Fallback function to estimate yield using rules when model is unavailable
    
    Args:
        input_data (dict): Environmental data with crop type
        
    Returns:
        dict: Estimated yield and confidence
    """
    # Extract parameters
    temperature = input_data.get('temperature', 25)
    humidity = input_data.get('humidity', 70)
    ph = input_data.get('ph', 6.5)
    rainfall = input_data.get('rainfall', 100)
    crop = input_data.get('crop', 'unknown')
    
    # Define base yields for known crops
    base_yields = {
        'rice': 5000,
        'wheat': 3500,
        'maize': 6000,
        'Sugarcane': 70000,
        'Cotton': 2500,
        'Jute': 2200,
        'Coffee': 1800,
        'Tea': 2000,
        'potato': 25000,
        'tomato': 35000,
        'banana': 30000,
        'apple': 25000,
        'mango': 15000,
        'grapes': 20000,
        'orange': 22000,
        'papaya': 30000,
        'watermelon': 35000
    }
    
    # Get base yield for the crop or use a default
    base_yield = base_yields.get(crop, 4000)
    
    # Define optimal ranges
    optimal_temp = (20, 30)
    optimal_humidity = (60, 80)
    optimal_ph = (5.5, 7.5)
    optimal_rainfall = (80, 300)
    
    # Calculate factor scores
    temp_score = calculate_factor_score(temperature, optimal_temp)
    humidity_score = calculate_factor_score(humidity, optimal_humidity)
    ph_score = calculate_factor_score(ph, optimal_ph)
    rainfall_score = calculate_factor_score(rainfall, optimal_rainfall)
    
    # Calculate overall score
    overall_score = (temp_score * 0.3 + humidity_score * 0.3 + ph_score * 0.2 + rainfall_score * 0.2)
    
    # Adjust base yield
    adjusted_yield = base_yield * (0.5 + overall_score)
    
    # Add variability
    final_yield = adjusted_yield * (0.9 + 0.2 * np.random.random())
    
    # Calculate confidence
    confidence = overall_score * 85  # Max 85% confidence for rule-based estimation
    
    # Return result
    result = {
        'predicted_yield': round(final_yield, 2),
        'confidence': round(confidence, 2),
        'lower_bound': round(final_yield * 0.8, 2),
        'upper_bound': round(final_yield * 1.2, 2),
        'unit': 'kg/hectare',
        'factors': {
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        },
        'note': 'Estimated using rule-based fallback method'
    }
    
    return result

if __name__ == "__main__":
    try:
        # Parse command-line arguments
        if len(sys.argv) >= 3:
            input_path = sys.argv[1]
            model_output_path = sys.argv[2]
        else:
            # Default paths
            input_path = os.path.join("data", "cpdata.csv")
            model_output_path = os.path.join("ml", "models", "yield_prediction_model.pkl")
        
        # Prepare dataset with yield information
        prepared_data_path = os.path.join("data", "yield_data.csv")
        yield_data = prepare_yield_dataset(input_path, prepared_data_path)
        
        # Train the model
        model, metrics = train_yield_model(yield_data, model_output_path)
        
        logger.info(f"Training completed successfully with accuracy: {metrics['accuracy']:.2f}%")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)