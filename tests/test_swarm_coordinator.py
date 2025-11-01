"""Unit tests for swarm coordinator."""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from swarm.coordinator import SwarmCoordinator
from edge.edge_node import EdgeNode
from models.classifier import DisasterClassifier


class TestSwarmCoordinator(unittest.TestCase):
    """Test cases for SwarmCoordinator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.coordinator = SwarmCoordinator()
        
        # Create nodes
        self.node1 = EdgeNode("node_1", (37.7749, -122.4194))
        self.node2 = EdgeNode("node_2", (34.0522, -118.2437))
        self.node3 = EdgeNode("node_3", (40.7128, -74.0060))
        
        # Create and train classifier
        np.random.seed(42)
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 3, 50)
        
        self.classifier = DisasterClassifier(n_estimators=5, max_depth=3)
        self.classifier.train(X_train, y_train)
    
    def test_initialization(self):
        """Test coordinator initialization."""
        self.assertEqual(len(self.coordinator.nodes), 0)
        self.assertGreater(self.coordinator.consensus_threshold, 0)
    
    def test_register_node(self):
        """Test node registration."""
        self.coordinator.register_node(self.node1)
        self.assertEqual(len(self.coordinator.nodes), 1)
        self.assertIn("node_1", self.coordinator.nodes)
    
    def test_unregister_node(self):
        """Test node unregistration."""
        self.coordinator.register_node(self.node1)
        self.coordinator.register_node(self.node2)
        
        self.coordinator.unregister_node("node_1")
        self.assertEqual(len(self.coordinator.nodes), 1)
        self.assertNotIn("node_1", self.coordinator.nodes)
    
    def test_broadcast_model(self):
        """Test model broadcasting."""
        self.coordinator.register_node(self.node1)
        self.coordinator.register_node(self.node2)
        
        self.coordinator.broadcast_model(self.classifier)
        
        self.assertIsNotNone(self.node1.classifier)
        self.assertIsNotNone(self.node2.classifier)
    
    def test_aggregate_predictions_empty(self):
        """Test aggregation with no predictions."""
        result = self.coordinator.aggregate_predictions([])
        self.assertIn('error', result)
    
    def test_aggregate_predictions_valid(self):
        """Test aggregation with valid predictions."""
        predictions = [
            {'prediction': 1, 'confidence': 0.8},
            {'prediction': 1, 'confidence': 0.9},
            {'prediction': 2, 'confidence': 0.7},
        ]
        
        result = self.coordinator.aggregate_predictions(predictions)
        
        self.assertIn('swarm_prediction', result)
        self.assertIn('consensus_strength', result)
        self.assertEqual(result['swarm_prediction'], 1)  # Majority is 1
    
    def test_process_distributed(self):
        """Test distributed processing."""
        self.coordinator.register_node(self.node1)
        self.coordinator.register_node(self.node2)
        self.coordinator.broadcast_model(self.classifier)
        
        sensor_readings = [
            {
                'urgency_level': 'High',
                'temperature_c': 45.0,
                'gas_ppm': 150,
                'humidity_percent': 20,
                'flood_sensor_level_cm': 0
            },
            {
                'urgency_level': 'Medium',
                'temperature_c': 30.0,
                'gas_ppm': 80,
                'humidity_percent': 60,
                'flood_sensor_level_cm': 0
            }
        ]
        
        result = self.coordinator.process_distributed(sensor_readings)
        
        self.assertNotIn('error', result)
        self.assertIn('swarm_prediction', result)
        self.assertIn('individual_predictions', result)
    
    def test_get_swarm_status(self):
        """Test swarm status retrieval."""
        self.coordinator.register_node(self.node1)
        self.coordinator.register_node(self.node2)
        
        status = self.coordinator.get_swarm_status()
        
        self.assertEqual(status['total_nodes'], 2)
        self.assertIn('node_1', status['node_ids'])
        self.assertIn('node_2', status['node_ids'])


if __name__ == '__main__':
    unittest.main()
