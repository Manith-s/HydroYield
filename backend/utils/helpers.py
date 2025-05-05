"""
Helper functions for the Smart Hydroponic System
"""

import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

def validate_sensor_data(data):
    """
    Validate sensor data from IoT devices
    
    Args:
        data (dict): Sensor data to validate
        
    Returns:
        dict: Validation result with 'valid' flag and 'errors' list
    """
    errors = []
    
    # Check if data is a dictionary
    if not isinstance(data, dict):
        return {'valid': False, 'errors': ['Data must be a JSON object']}
    
    # Check required fields
    if 'device_id' not in data:
        errors.append('Missing device_id')
    
    # Validate temperature
    if 'temperature' in data:
        temp = data['temperature']
        if not isinstance(temp, (int, float)):
            errors.append('Temperature must be a number')
        elif temp < -50 or temp > 100:
            errors.append('Temperature out of reasonable range (-50 to 100°C)')
    
    # Validate humidity
    if 'humidity' in data:
        humidity = data['humidity']
        if not isinstance(humidity, (int, float)):
            errors.append('Humidity must be a number')
        elif humidity < 0 or humidity > 100:
            errors.append('Humidity out of range (0-100%)')
    
    # Validate pH
    if 'ph' in data:
        ph = data['ph']
        if not isinstance(ph, (int, float)):
            errors.append('pH must be a number')
        elif ph < 0 or ph > 14:
            errors.append('pH out of range (0-14)')
    
    # Validate timestamp if provided
    if 'timestamp' in data and isinstance(data['timestamp'], str):
        try:
            datetime.fromisoformat(data['timestamp'])
        except ValueError:
            errors.append('Invalid timestamp format (use ISO format)')
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def validate_prediction_input(data):
    """
    Validate input data for crop predictions
    
    Args:
        data (dict): Input data to validate
        
    Returns:
        dict: Validation result with 'valid' flag and 'errors' list
    """
    errors = []
    
    # Check if data is a dictionary
    if not isinstance(data, dict):
        return {'valid': False, 'errors': ['Data must be a JSON object']}
    
    # Required parameters
    required_params = ['temperature', 'humidity', 'ph']
    
    # Check required parameters
    for param in required_params:
        if param not in data:
            errors.append(f'Missing required parameter: {param}')
        elif not isinstance(data[param], (int, float)):
            errors.append(f'{param} must be a number')
    
    # Specific validations
    if 'temperature' in data and isinstance(data['temperature'], (int, float)):
        if data['temperature'] < -50 or data['temperature'] > 100:
            errors.append('Temperature out of reasonable range (-50 to 100°C)')
    
    if 'humidity' in data and isinstance(data['humidity'], (int, float)):
        if data['humidity'] < 0 or data['humidity'] > 100:
            errors.append('Humidity out of range (0-100%)')
    
    if 'ph' in data and isinstance(data['ph'], (int, float)):
        if data['ph'] < 0 or data['ph'] > 14:
            errors.append('pH out of range (0-14)')
    
    return {
        'valid': len(errors) == 0,
        'errors': errors
    }

def format_timestamp(timestamp):
    """
    Format timestamp for display
    
    Args:
        timestamp: Timestamp to format
        
    Returns:
        str: Formatted timestamp
    """
    if isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp)
        except ValueError:
            return timestamp
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        return str(timestamp)
    
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def sanitize_input(input_str):
    """
    Sanitize input string to prevent injection attacks
    
    Args:
        input_str (str): Input string to sanitize
        
    Returns:
        str: Sanitized string
    """
    if not isinstance(input_str, str):
        return str(input_str)
    
    # Remove any potentially harmful characters
    sanitized = re.sub(r'[;<>$|&]', '', input_str)
    return sanitized.strip()

def send_notification(subject, message, recipient=None):
    """
    Send notification about important system events
    
    Args:
        subject (str): Notification subject
        message (str): Notification message
        recipient (str, optional): Recipient email
        
    Returns:
        bool: Success flag
    """
    from flask import current_app
    import smtplib
    from email.mime.text import MIMEText
    
    # Check if notifications are enabled
    if not current_app.config.get('ENABLE_NOTIFICATIONS', False):
        logger.info(f"Notification not sent (disabled): {subject}")
        return False
    
    try:
        # Get recipient
        recipient = recipient or current_app.config.get('NOTIFICATION_EMAIL')
        if not recipient:
            logger.warning("No recipient specified for notification")
            return False
        
        # Get SMTP settings
        smtp_server = current_app.config.get('SMTP_SERVER')
        smtp_port = current_app.config.get('SMTP_PORT')
        smtp_username = current_app.config.get('SMTP_USERNAME')
        smtp_password = current_app.config.get('SMTP_PASSWORD')
        
        # Create message
        msg = MIMEText(message)
        msg['Subject'] = f"Hydroponic System: {subject}"
        msg['From'] = smtp_username
        msg['To'] = recipient
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        
        logger.info(f"Notification sent: {subject}")
        return True
        
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        return False