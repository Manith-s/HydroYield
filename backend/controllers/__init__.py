
"""
Controllers package initialization
"""

from .sensor_controller import sensor_bp
from .prediction_controller import prediction_bp

# Export blueprints
__all__ = [
    'sensor_bp',
    'prediction_bp'
]