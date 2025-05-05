"""
Controller for handling sensor data from IoT devices
"""

import logging
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import json
from services.data_service import process_sensor_data, get_sensor_data_history
from utils.helpers import validate_sensor_data

# Create Blueprint
sensor_bp = Blueprint('sensor', __name__)
logger = logging.getLogger(__name__)

@sensor_bp.route('/sensor-data', methods=['POST'])
def receive_sensor_data():
    """
    Endpoint to receive sensor data from IoT devices
    """
    try:
        # Get data from request
        data = request.get_json()
        
        # Basic validation
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        # Validate sensor data
        validation_result = validate_sensor_data(data)
        if not validation_result['valid']:
            return jsonify({
                'status': 'error',
                'message': 'Invalid sensor data',
                'details': validation_result['errors']
            }), 400
        
        # Add timestamp if not provided
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
        
        # Process the sensor data
        processed_data = process_sensor_data(data)
        
        # Store in database
        db = current_app.config['db'] if 'db' in current_app.config else None
        if db:
            collection = db[current_app.config['IOT_DATA_COLLECTION']]
            collection.insert_one(processed_data)
            logger.info(f"Stored sensor data for device {processed_data.get('device_id', 'unknown')}")
        
        return jsonify({
            'status': 'success',
            'message': 'Sensor data received and processed',
            'data': processed_data
        })
        
    except Exception as e:
        logger.error(f"Error processing sensor data: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to process sensor data',
            'error': str(e)
        }), 500

@sensor_bp.route('/sensor-data/<device_id>', methods=['GET'])
def get_device_data(device_id):
    """
    Endpoint to retrieve sensor data history for a specific device
    """
    try:
        # Get query parameters
        days = request.args.get('days', default=1, type=int)
        limit = request.args.get('limit', default=100, type=int)
        
        # Get data from service
        data = get_sensor_data_history(device_id, days, limit)
        
        return jsonify({
            'status': 'success',
            'device_id': device_id,
            'data': data
        })
        
    except Exception as e:
        logger.error(f"Error retrieving sensor data: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to retrieve sensor data',
            'error': str(e)
        }), 500

@sensor_bp.route('/devices', methods=['GET'])
def get_devices():
    """
    Endpoint to get a list of all registered devices
    """
    try:
        db = current_app.config['db'] if 'db' in current_app.config else None
        if not db:
            return jsonify({
                'status': 'error',
                'message': 'Database not available'
            }), 500
        
        collection = db[current_app.config['IOT_DATA_COLLECTION']]
        
        # Get unique device IDs
        devices = collection.distinct('device_id')
        
        # Get last reading for each device
        device_info = []
        for device_id in devices:
            last_reading = collection.find({'device_id': device_id}).sort('timestamp', -1).limit(1)
            
            if last_reading.count() > 0:
                reading = last_reading[0]
                device_info.append({
                    'device_id': device_id,
                    'last_reading_time': reading.get('timestamp', ''),
                    'status': 'active' if (datetime.now() - datetime.fromisoformat(reading.get('timestamp'))).total_seconds() < 3600 else 'inactive'
                })
        
        return jsonify({
            'status': 'success',
            'count': len(device_info),
            'devices': device_info
        })
        
    except Exception as e:
        logger.error(f"Error retrieving devices: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to retrieve devices',
            'error': str(e)
        }), 500