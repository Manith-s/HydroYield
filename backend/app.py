"""
Main Flask application for Smart Hydroponic System
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pymongo
import logging
from datetime import datetime

# Import configuration
from config import get_config

# Import controllers
from controllers.sensor_controller import sensor_bp
from controllers.prediction_controller import prediction_bp

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load configuration based on environment
config = get_config()
app.config.from_object(config)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize MongoDB client
try:
    mongo_client = pymongo.MongoClient(config.DB_URI)
    db = mongo_client[config.DB_NAME]
    logger.info(f"Connected to MongoDB: {config.DB_URI}")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    mongo_client = None
    db = None

# Register blueprints
app.register_blueprint(sensor_bp, url_prefix='/api')
app.register_blueprint(prediction_bp, url_prefix='/api')

# Default route
@app.route('/')
def index():
    return jsonify({
        'status': 'success',
        'message': 'Smart Hydroponic System API is running',
        'timestamp': datetime.now().isoformat()
    })

# Health check route
@app.route('/health')
def health():
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database': 'connected' if mongo_client else 'disconnected'
    }
    return jsonify(health_status)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Resource not found',
        'error': str(error)
    }), 404

@app.errorhandler(500)
def server_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'error': str(error)
    }), 500

# Custom middleware to verify API key
@app.before_request
def verify_api_key():
    # Skip API key verification for certain paths
    if request.path in ['/', '/health'] or request.method == 'OPTIONS':
        return None
    
    api_key = request.headers.get('X-API-Key')
    if api_key != app.config['API_KEY']:
        return jsonify({
            'status': 'error',
            'message': 'Invalid or missing API key'
        }), 401

# Context processor for database access
@app.context_processor
def inject_db():
    return dict(db=db)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)