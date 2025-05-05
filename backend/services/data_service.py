"""
Service for processing and managing sensor data
"""

import logging
from datetime import datetime, timedelta
from flask import current_app
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def process_sensor_data(data):
    """
    Process incoming sensor data:
    - Validate and clean
    - Normalize values if needed
    - Add metadata
    
    Args:
        data (dict): Raw sensor data
        
    Returns:
        dict: Processed data
    """
    processed_data = data.copy()
    
    # Convert timestamp to datetime if it's a string
    if 'timestamp' in processed_data and isinstance(processed_data['timestamp'], str):
        try:
            # Parse the timestamp
            processed_data['timestamp'] = datetime.fromisoformat(processed_data['timestamp'])
        except ValueError:
            # If parsing fails, use current time
            processed_data['timestamp'] = datetime.now()
    
    # Ensure we have required fields
    required_fields = ['temperature', 'humidity', 'ph']
    for field in required_fields:
        if field not in processed_data:
            processed_data[field] = None
    
    # Add data quality metrics
    processed_data['data_quality'] = calculate_data_quality(processed_data)
    
    # Add calculated fields like VPD (Vapor Pressure Deficit)
    if processed_data['temperature'] is not None and processed_data['humidity'] is not None:
        processed_data['vpd'] = calculate_vpd(
            processed_data['temperature'], 
            processed_data['humidity']
        )
    
    # Add system metadata
    processed_data['processing_timestamp'] = datetime.now()
    
    return processed_data

def get_sensor_data_history(device_id, days=1, limit=100):
    """
    Retrieve sensor data history for a specific device
    
    Args:
        device_id (str): Device identifier
        days (int): Number of days to look back
        limit (int): Maximum number of records to return
        
    Returns:
        list: History of sensor readings
    """
    try:
        db = current_app.config['db'] if 'db' in current_app.config else None
        if not db:
            logger.error("Database not available")
            return []
        
        collection = db[current_app.config['IOT_DATA_COLLECTION']]
        
        # Calculate the date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Query the database
        cursor = collection.find({
            'device_id': device_id,
            'timestamp': {'$gte': start_date, '$lte': end_date}
        }).sort('timestamp', -1).limit(limit)
        
        # Convert to list and process dates for JSON serialization
        history = []
        for doc in cursor:
            # Convert ObjectId to string for JSON serialization
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
            
            # Convert datetime objects to ISO format strings
            for key, value in doc.items():
                if isinstance(value, datetime):
                    doc[key] = value.isoformat()
            
            history.append(doc)
        
        return history
        
    except Exception as e:
        logger.error(f"Error retrieving sensor data history: {e}")
        return []

def calculate_data_quality(data):
    """
    Calculate data quality score based on sensor readings
    
    Args:
        data (dict): Sensor data
        
    Returns:
        float: Quality score (0-1)
    """
    # Define normal ranges for each parameter
    normal_ranges = {
        'temperature': (0, 50),  # 0-50°C
        'humidity': (0, 100),    # 0-100%
        'ph': (0, 14)            # 0-14 pH
    }
    
    # Calculate quality score
    valid_params = 0
    total_params = 0
    
    for param, (min_val, max_val) in normal_ranges.items():
        if param in data and data[param] is not None:
            total_params += 1
            if min_val <= data[param] <= max_val:
                valid_params += 1
    
    # Return quality score (ratio of valid parameters)
    return valid_params / total_params if total_params > 0 else 0

def calculate_vpd(temperature, humidity):
    """
    Calculate Vapor Pressure Deficit (VPD)
    
    Args:
        temperature (float): Temperature in Celsius
        humidity (float): Relative humidity (0-100)
        
    Returns:
        float: VPD in kPa
    """
    # Saturated vapor pressure calculation (kPa)
    svp = 0.61078 * np.exp((17.27 * temperature) / (temperature + 237.3))
    
    # Actual vapor pressure
    avp = svp * (humidity / 100.0)
    
    # Vapor pressure deficit
    vpd = svp - avp
    
    return max(0, vpd)

def get_daily_averages(device_id, days=7):
    """
    Calculate daily averages for sensor readings
    
    Args:
        device_id (str): Device identifier
        days (int): Number of days to look back
        
    Returns:
        dict: Daily averages for each parameter
    """
    try:
        # Get sensor history
        history = get_sensor_data_history(device_id, days, 1000)
        
        # Convert to DataFrame
        df = pd.DataFrame(history)
        
        # Check if we have any data
        if df.empty:
            return {}
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Set timestamp as index and resample by day
        df.set_index('timestamp', inplace=True)
        daily_avg = df.resample('D').mean()
        
        # Convert back to dictionary format
        result = {}
        for day, row in daily_avg.iterrows():
            day_str = day.strftime('%Y-%m-%d')
            result[day_str] = row.to_dict()
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating daily averages: {e}")
        return {}