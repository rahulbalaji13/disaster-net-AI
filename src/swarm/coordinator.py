"""
Swarm coordination system for distributed IoT disaster response.
"""
import numpy as np
from typing import List, Dict, Any, Optional


class SwarmCoordinator:
    """
    Coordinates multiple edge nodes in a swarm intelligence system.
    Implements consensus mechanisms and distributed decision making.
    """
    
    def __init__(self):
        """Initialize swarm coordinator."""
        self.nodes = {}
        self.consensus_threshold = 0.7
        
    def register_node(self, node):
        """Register an edge node with the swarm."""
        self.nodes[node.node_id] = node
        
    def unregister_node(self, node_id: str):
        """Remove a node from the swarm."""
        if node_id in self.nodes:
            del self.nodes[node_id]
    
    def broadcast_model(self, classifier):
        """
        Broadcast a trained model to all nodes in the swarm.
        
        Args:
            classifier: Trained classifier to distribute
        """
        for node in self.nodes.values():
            node.set_classifier(classifier)
    
    def aggregate_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate predictions from multiple nodes using voting.
        
        Args:
            predictions: List of prediction dictionaries from nodes
            
        Returns:
            Aggregated consensus prediction
        """
        if not predictions:
            return {'error': 'No predictions to aggregate'}
        
        # Extract prediction values and confidences
        pred_values = [p['prediction'] for p in predictions if 'prediction' in p]
        confidences = [p['confidence'] for p in predictions if 'confidence' in p]
        
        if not pred_values:
            return {'error': 'No valid predictions'}
        
        # Majority voting
        unique_preds, counts = np.unique(pred_values, return_counts=True)
        majority_pred = unique_preds[np.argmax(counts)]
        majority_count = np.max(counts)
        
        # Calculate consensus strength
        consensus = majority_count / len(pred_values)
        
        # Average confidence for majority prediction
        majority_confidences = [
            confidences[i] for i, p in enumerate(pred_values) 
            if p == majority_pred
        ]
        avg_confidence = np.mean(majority_confidences) if majority_confidences else 0.0
        
        return {
            'swarm_prediction': int(majority_pred),
            'consensus_strength': float(consensus),
            'average_confidence': float(avg_confidence),
            'total_nodes': len(predictions),
            'voting_nodes': len(pred_values),
            'high_consensus': consensus >= self.consensus_threshold
        }
    
    def process_distributed(self, sensor_readings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process sensor readings across the swarm in a distributed manner.
        
        Args:
            sensor_readings: List of sensor reading dictionaries
            
        Returns:
            Swarm-level aggregated result
        """
        if not self.nodes:
            return {'error': 'No nodes registered in swarm'}
        
        predictions = []
        node_list = list(self.nodes.values())
        
        # Distribute readings across nodes
        for i, reading in enumerate(sensor_readings):
            node = node_list[i % len(node_list)]
            result = node.process_sensor_data(reading)
            if 'prediction' in result:
                predictions.append(result)
        
        # Aggregate results
        aggregated = self.aggregate_predictions(predictions)
        aggregated['individual_predictions'] = predictions
        
        return aggregated
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get status of the entire swarm."""
        return {
            'total_nodes': len(self.nodes),
            'node_ids': list(self.nodes.keys()),
            'consensus_threshold': self.consensus_threshold
        }
