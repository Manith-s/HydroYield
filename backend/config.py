"""
Configuration settings for the Smart Hydroponic System
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Application settings
class Config:
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    DEBUG = os.getenv('DEBUG', 'False') == 'True'
    
    # Database settings
    DB_URI = os.getenv('DB_URI', 'mongodb://localhost:27017/')
    DB_NAME = os.getenv('DB_NAME', 'hydroponic_system')
    
    # API settings
    API_KEY = os.getenv('API_KEY', 'your-api-key-here')
    
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', 'ml/models/random_forest_model.pkl')
    
    # IoT settings
    IOT_DATA_COLLECTION = 'sensor_data'
    
    # Crop data settings
    CROP_DATA_PATH = os.getenv('CROP_DATA_PATH', 'data/cpdata.csv')
    
    # Notification settings
    ENABLE_NOTIFICATIONS = os.getenv('ENABLE_NOTIFICATIONS', 'False') == 'True'
    NOTIFICATION_EMAIL = os.getenv('NOTIFICATION_EMAIL', '')
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    SMTP_USERNAME = os.getenv('SMTP_USERNAME', '')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
    
    # Optimal growing parameters
    OPTIMAL_PARAMS = {
        'rice': {
            'temperature': {'min': 20, 'max': 27},
            'humidity': {'min': 80, 'max': 90},
            'ph': {'min': 6.0, 'max': 7.5},
        },
        'wheat': {
            'temperature': {'min': 15, 'max': 24},
            'humidity': {'min': 60, 'max': 70},
            'ph': {'min': 5.5, 'max': 7.0},
        },
        'maize': {
            'temperature': {'min': 21, 'max': 32},
            'humidity': {'min': 65, 'max': 85},
            'ph': {'min': 5.8, 'max': 7.0},
        },
        # Add more crops as needed
    }

# Development configuration
class DevelopmentConfig(Config):
    DEBUG = True

# Production configuration
class ProductionConfig(Config):
    DEBUG = False

# Testing configuration
class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    DB_NAME = os.getenv('TEST_DB_NAME', 'hydroponic_system_test')

# Configuration dictionary
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}

# Get current configuration
def get_config():
    env = os.getenv('FLASK_ENV', 'development')
    return config_by_name[env]