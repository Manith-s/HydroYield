"""
Service for machine learning predictions and analysis
"""

import logging
import os
import joblib
import pandas as pd
import numpy as np
from flask import current_app
from sklearn.ensemble import RandomForestClassifier

# Import yield prediction functionality
from ml.crop_yield_predictor import predict_yield, estimate_yield_rule_based

logger = logging.getLogger(__name__)

# Load the trained model
def load_model():
    """
    Load the trained Random Forest model
    
    Returns:
        object: Trained model
    """
    try:
        model_path = current_app.config.get('MODEL_PATH')
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path}")
            return None
        
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def predict_crop(data):
    """
    Predict suitable crop based on environmental parameters
    
    Args:
        data (dict): Environmental data with temperature, humidity, pH, etc.
        
    Returns:
        dict: Prediction results with crop, confidence, and alternatives
    """
    try:
        # Load model
        model = load_model()
        if model is None:
            # If model is not available, use rule-based prediction
            return rule_based_prediction(data)
        
        # Prepare input data
        input_data = {
            'temperature': data.get('temperature', 0),
            'humidity': data.get('humidity', 0),
            'ph': data.get('ph', 0),
            'rainfall': data.get('rainfall', 0)
        }
        
        # Convert to DataFrame for prediction
        df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(df)[0]
        confidence = np.max(probabilities) * 100
        
        # Get top alternative crops
        classes = model.classes_
        top_indices = np.argsort(probabilities)[::-1][:5]  # Top 5 crops
        alternatives = [
            {'crop': classes[i], 'confidence': probabilities[i] * 100} 
            for i in top_indices[1:]  # Skip the top one as it's the main prediction
        ]
        
        return {
            'crop': prediction,
            'confidence': confidence,
            'alternatives': alternatives
        }
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return rule_based_prediction(data)

def predict_crop_yield(data):
    """
    Predict crop yield based on environmental parameters and crop type
    
    Args:
        data (dict): Environmental data and crop type
        
    Returns:
        dict: Yield prediction with confidence and range
    """
    try:
        # Get yield model path
        yield_model_path = os.path.join(
            os.path.dirname(current_app.config.get('MODEL_PATH')),
            'yield_prediction_model.pkl'
        )
        
        # Check if model exists
        if os.path.exists(yield_model_path):
            # Use the model for prediction
            yield_prediction = predict_yield(yield_model_path, data)
        else:
            # Use rule-based estimation
            logger.warning("Yield prediction model not found, using rule-based estimation")
            yield_prediction = estimate_yield_rule_based(data)
        
        return yield_prediction
    
    except Exception as e:
        logger.error(f"Error predicting crop yield: {e}")
        
        # Fallback to rule-based estimation
        return estimate_yield_rule_based(data)

def rule_based_prediction(data):
    """
    Rule-based crop prediction when ML model is not available
    
    Args:
        data (dict): Environmental data
        
    Returns:
        dict: Prediction results
    """
    # Get parameters
    temperature = data.get('temperature', 0)
    humidity = data.get('humidity', 0)
    ph = data.get('ph', 0)
    rainfall = data.get('rainfall', 0)
    
    # Simple rule-based prediction
    optimal_params = current_app.config.get('OPTIMAL_PARAMS', {})
    scores = {}
    
    for crop, params in optimal_params.items():
        # Calculate score based on parameter proximity to optimal ranges
        temp_score = score_in_range(temperature, params['temperature']['min'], params['temperature']['max'])
        humid_score = score_in_range(humidity, params['humidity']['min'], params['humidity']['max'])
        ph_score = score_in_range(ph, params['ph']['min'], params['ph']['max'])
        
        # Weighted average of scores
        total_score = (temp_score * 0.4 + humid_score * 0.3 + ph_score * 0.3) * 100
        scores[crop] = total_score
    
    # Sort by score
    sorted_crops = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get top crop and alternatives
    if not sorted_crops:
        return {
            'crop': 'unknown',
            'confidence': 0,
            'alternatives': []
        }
    
    top_crop, top_score = sorted_crops[0]
    alternatives = [{'crop': crop, 'confidence': score} for crop, score in sorted_crops[1:5]]
    
    return {
        'crop': top_crop,
        'confidence': top_score,
        'alternatives': alternatives
    }

def score_in_range(value, min_val, max_val):
    """
    Calculate score (0-1) based on how close a value is to the optimal range
    
    Args:
        value (float): Value to evaluate
        min_val (float): Minimum optimal value
        max_val (float): Maximum optimal value
        
    Returns:
        float: Score between 0 and 1
    """
    if min_val <= value <= max_val:
        # Value is in optimal range
        return 1.0
    
    # Calculate distance from range
    if value < min_val:
        distance = min_val - value
        reference = min_val
    else:
        distance = value - max_val
        reference = max_val
    
    # Score decreases with distance from range
    return max(0, 1 - (distance / (reference * 0.5)))

def get_optimal_conditions(crop):
    """
    Get optimal growing conditions for a specific crop
    
    Args:
        crop (str): Crop name
        
    Returns:
        dict: Optimal conditions or None if crop not found
    """
    optimal_params = current_app.config.get('OPTIMAL_PARAMS', {})
    return optimal_params.get(crop.lower(), None)

def suggest_improvements(crop, current_conditions):
    """
    Suggest improvements based on current conditions and optimal ranges
    
    Args:
        crop (str): Crop name
        current_conditions (dict): Current environmental conditions
        
    Returns:
        dict: Suggested improvements for each parameter
    """
    # Get optimal conditions
    optimal = get_optimal_conditions(crop)
    if not optimal:
        return {'error': f'No data available for crop: {crop}'}
    
    suggestions = {}
    
    # Check each parameter
    for param, value in current_conditions.items():
        if param in optimal:
            min_val = optimal[param]['min']
            max_val = optimal[param]['max']
            
            if value < min_val:
                suggestions[param] = f"Increase {param} from {value} to at least {min_val}"
            elif value > max_val:
                suggestions[param] = f"Decrease {param} from {value} to at most {max_val}"
            else:
                suggestions[param] = f"{param} is in optimal range"
    
    return suggestions

def analyze_crop_data(crop_name, time_range=None):
    """
    Analyze historical data for a specific crop
    
    Args:
        crop_name (str): Crop name to analyze
        time_range (tuple, optional): Start and end dates
        
    Returns:
        dict: Analysis results
    """
    try:
        # Get database connection
        db = current_app.config.get('db')
        if not db:
            return {'error': 'Database not available'}
        
        # Query conditions
        query = {'crop': crop_name}
        if time_range:
            start_date, end_date = time_range
            query['timestamp'] = {
                '$gte': start_date,
                '$lte': end_date
            }
        
        # Get data from database
        collection = db['sensor_readings']
        cursor = collection.find(query)
        
        # Convert to DataFrame
        readings = list(cursor)
        if not readings:
            return {'message': f'No data found for crop: {crop_name}'}
        
        df = pd.DataFrame(readings)
        
        # Basic statistics
        stats = {}
        for param in ['temperature', 'humidity', 'ph', 'rainfall']:
            if param in df.columns:
                stats[param] = {
                    'min': df[param].min(),
                    'max': df[param].max(),
                    'avg': df[param].mean(),
                    'std': df[param].std()
                }
        
        # Get optimal conditions
        optimal = get_optimal_conditions(crop_name)
        
        # Calculate time in optimal range for each parameter
        optimal_time = {}
        if optimal:
            for param, ranges in optimal.items():
                if param in df.columns:
                    min_val = ranges['min']
                    max_val = ranges['max']
                    in_range = ((df[param] >= min_val) & (df[param] <= max_val)).mean() * 100
                    optimal_time[param] = in_range
        
        # Return analysis results
        return {
            'crop': crop_name,
            'sample_size': len(df),
            'statistics': stats,
            'optimal_conditions': optimal,
            'time_in_optimal_range': optimal_time
        }
        
    except Exception as e:
        logger.error(f"Error analyzing crop data: {e}")
        return {'error': f'Analysis failed: {str(e)}'}