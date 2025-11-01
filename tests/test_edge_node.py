"""Unit tests for edge node."""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from edge.edge_node import EdgeNode
from models.classifier import DisasterClassifier


class TestEdgeNode(unittest.TestCase):
    """Test cases for EdgeNode class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.node = EdgeNode("test_node_1", (37.7749, -122.4194))
        
        # Create and train a simple classifier
        np.random.seed(42)
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 3, 50)
        
        self.classifier = DisasterClassifier(n_estimators=5, max_depth=3)
        self.classifier.train(X_train, y_train)
    
    def test_initialization(self):
        """Test node initialization."""
        self.assertEqual(self.node.node_id, "test_node_1")
        self.assertEqual(self.node.location, (37.7749, -122.4194))
        self.assertIsNone(self.node.classifier)
    
    def test_set_classifier(self):
        """Test setting classifier."""
        self.node.set_classifier(self.classifier)
        self.assertIsNotNone(self.node.classifier)
    
    def test_process_sensor_data_without_classifier(self):
        """Test processing without classifier."""
        sensor_reading = {
            'urgency_level': 'High',
            'temperature_c': 45.0,
            'gas_ppm': 150,
            'humidity_percent': 20,
            'flood_sensor_level_cm': 0
        }
        
        result = self.node.process_sensor_data(sensor_reading)
        
        self.assertEqual(result['processing_type'], 'raw')
        self.assertIn('note', result)
    
    def test_process_sensor_data_with_classifier(self):
        """Test processing with classifier."""
        self.node.set_classifier(self.classifier)
        
        sensor_reading = {
            'urgency_level': 'High',
            'temperature_c': 45.0,
            'gas_ppm': 150,
            'humidity_percent': 20,
            'flood_sensor_level_cm': 0
        }
        
        result = self.node.process_sensor_data(sensor_reading)
        
        self.assertEqual(result['processing_type'], 'edge')
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        self.assertGreater(result['confidence'], 0)
    
    def test_sensor_cache(self):
        """Test sensor data caching."""
        sensor_reading = {
            'urgency_level': 'Medium',
            'temperature_c': 30.0,
            'gas_ppm': 80,
            'humidity_percent': 50,
            'flood_sensor_level_cm': 0
        }
        
        # Process multiple readings
        for _ in range(5):
            self.node.process_sensor_data(sensor_reading)
        
        self.assertEqual(len(self.node.sensor_cache), 5)
    
    def test_get_statistics(self):
        """Test node statistics."""
        stats = self.node.get_statistics()
        
        self.assertEqual(stats['node_id'], 'test_node_1')
        self.assertEqual(stats['location'], (37.7749, -122.4194))
        self.assertIn('cached_readings', stats)
        self.assertIn('has_classifier', stats)


if __name__ == '__main__':
    unittest.main()
