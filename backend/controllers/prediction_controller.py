"""
Controller for handling crop predictions
"""

import logging
from flask import Blueprint, request, jsonify, current_app
from services.ml_service import predict_crop, predict_crop_yield, get_optimal_conditions, suggest_improvements, analyze_crop_data
from utils.helpers import validate_prediction_input

# Create Blueprint
prediction_bp = Blueprint('prediction', __name__)
logger = logging.getLogger(__name__)

@prediction_bp.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict suitable crops based on environmental conditions
    """
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate input data
        validation_result = validate_prediction_input(data)
        if not validation_result['valid']:
            return jsonify({
                'status': 'error',
                'message': 'Invalid input data',
                'details': validation_result['errors']
            }), 400
        
        # Make prediction
        prediction_result = predict_crop(data)
        
        return jsonify({
            'status': 'success',
            'prediction': prediction_result['crop'],
            'confidence': prediction_result['confidence'],
            'suitable_crops': prediction_result['alternatives']
        })
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to make prediction',
            'error': str(e)
        }), 500

@prediction_bp.route('/predict-yield', methods=['POST'])
def predict_yield():
    """
    Endpoint to predict crop yield based on environmental conditions and crop type
    """
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate input data
        if not data or 'crop' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required parameters. Must include crop type.'
            }), 400
        
        # Basic parameter validation
        for param in ['temperature', 'humidity', 'ph', 'rainfall']:
            if param not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required parameter: {param}'
                }), 400
            
            value = data[param]
            if not isinstance(value, (int, float)):
                return jsonify({
                    'status': 'error',
                    'message': f'Parameter {param} must be a number'
                }), 400
        
        # Make yield prediction
        yield_prediction = predict_crop_yield(data)
        
        # Enhance response with additional information
        response = {
            'status': 'success',
            'crop': data['crop'],
            'predicted_yield': yield_prediction['predicted_yield'],
            'unit': yield_prediction['unit'],
            'confidence': yield_prediction['confidence'],
            'yield_range': {
                'min': yield_prediction['lower_bound'],
                'max': yield_prediction['upper_bound']
            },
            'environmental_factors': yield_prediction['factors']
        }
        
        # Add note if present
        if 'note' in yield_prediction:
            response['note'] = yield_prediction['note']
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error predicting yield: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to predict yield',
            'error': str(e)
        }), 500

@prediction_bp.route('/optimal-conditions/<crop>', methods=['GET'])
def get_conditions(crop):
    """
    Endpoint to get optimal growing conditions for a specific crop
    """
    try:
        # Get optimal conditions from service
        conditions = get_optimal_conditions(crop)
        
        if not conditions:
            return jsonify({
                'status': 'error',
                'message': f'No data available for crop: {crop}'
            }), 404
        
        return jsonify({
            'status': 'success',
            'crop': crop,
            'optimal_conditions': conditions
        })
        
    except Exception as e:
        logger.error(f"Error retrieving optimal conditions: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to retrieve optimal conditions',
            'error': str(e)
        }), 500

@prediction_bp.route('/suggest-improvements', methods=['POST'])
def improvements():
    """
    Endpoint to suggest improvements for current conditions
    """
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate input data
        if not data or 'current_conditions' not in data or 'crop' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required data'
            }), 400
        
        # Get suggestions
        suggestions = suggest_improvements(data['crop'], data['current_conditions'])
        
        return jsonify({
            'status': 'success',
            'crop': data['crop'],
            'current_conditions': data['current_conditions'],
            'suggestions': suggestions
        })
        
    except Exception as e:
        logger.error(f"Error generating suggestions: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to generate suggestions',
            'error': str(e)
        }), 500

@prediction_bp.route('/analyze-crop/<crop_name>', methods=['GET'])
def analyze_crop(crop_name):
    """
    Endpoint to analyze historical data for a specific crop
    """
    try:
        # Get query parameters
        days = request.args.get('days', default=30, type=int)
        
        # Get analysis results
        results = analyze_crop_data(crop_name)
        
        return jsonify({
            'status': 'success',
            'crop': crop_name,
            'analysis': results
        })
        
    except Exception as e:
        logger.error(f"Error analyzing crop data: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to analyze crop data',
            'error': str(e)
        }), 500

@prediction_bp.route('/crops', methods=['GET'])
def get_crops():
    """
    Endpoint to get a list of all supported crops
    """
    try:
        # Get optimal conditions from config
        optimal_params = current_app.config.get('OPTIMAL_PARAMS', {})
        crops = list(optimal_params.keys())
        
        return jsonify({
            'status': 'success',
            'count': len(crops),
            'crops': crops
        })
        
    except Exception as e:
        logger.error(f"Error retrieving crops list: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to retrieve crops list',
            'error': str(e)
        }), 500