"""
Edge intelligence module for local AI processing on IoT devices.
"""
import numpy as np
from typing import Dict, Any, Optional


class EdgeNode:
    """
    Represents an edge computing node for disaster response.
    Performs local processing to reduce latency and bandwidth.
    """
    
    def __init__(self, node_id: str, location: tuple):
        """
        Initialize edge node.
        
        Args:
            node_id: Unique identifier for the node
            location: (latitude, longitude) tuple
        """
        self.node_id = node_id
        self.location = location
        self.classifier = None
        self.sensor_cache = []
        self.max_cache_size = 100
        
    def set_classifier(self, classifier):
        """Attach a trained classifier to this edge node."""
        self.classifier = classifier
        
    def process_sensor_data(self, sensor_reading: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process sensor data locally on the edge node.
        
        Args:
            sensor_reading: Dictionary with sensor values
            
        Returns:
            Processing result with classification
        """
        # Cache sensor reading
        self.sensor_cache.append(sensor_reading)
        if len(self.sensor_cache) > self.max_cache_size:
            self.sensor_cache.pop(0)
        
        # Extract features
        features = self._extract_features(sensor_reading)
        
        # Classify if model is available
        if self.classifier and self.classifier.is_trained:
            prediction = self.classifier.predict(features.reshape(1, -1))[0]
            probabilities = self.classifier.predict_proba(features.reshape(1, -1))[0]
            
            return {
                'node_id': self.node_id,
                'location': self.location,
                'prediction': int(prediction),
                'confidence': float(np.max(probabilities)),
                'sensor_reading': sensor_reading,
                'processing_type': 'edge'
            }
        else:
            return {
                'node_id': self.node_id,
                'location': self.location,
                'sensor_reading': sensor_reading,
                'processing_type': 'raw',
                'note': 'No classifier available, forwarding raw data'
            }
    
    def _extract_features(self, sensor_reading: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector from sensor reading."""
        urgency_map = {'Low': 0, 'Medium': 1, 'High': 2}
        
        features = np.array([
            urgency_map.get(sensor_reading.get('urgency_level', 'Low'), 0),
            sensor_reading.get('temperature_c', 0),
            sensor_reading.get('gas_ppm', 0),
            sensor_reading.get('humidity_percent', 0),
            sensor_reading.get('flood_sensor_level_cm', 0)
        ], dtype=float)
        
        return features
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get node statistics."""
        return {
            'node_id': self.node_id,
            'location': self.location,
            'cached_readings': len(self.sensor_cache),
            'has_classifier': self.classifier is not None
        }
