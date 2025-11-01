#!/usr/bin/env python3
"""
Demo script for DisasterNet SwarmAI system.
Demonstrates edge intelligence and swarm coordination for disaster response.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.classifier import DisasterClassifier
from utils.preprocessor import DataPreprocessor
from edge.edge_node import EdgeNode
from swarm.coordinator import SwarmCoordinator
import numpy as np


def main():
    """Run the DisasterNet SwarmAI demo."""
    print("=" * 60)
    print("DisasterNet SwarmAI - Disaster Response Demo")
    print("=" * 60)
    
    # Step 1: Load and preprocess data
    print("\n[1/5] Loading and preprocessing disaster data...")
    preprocessor = DataPreprocessor()
    data_path = 'DATASET/disaster_multimodal_dataset_10000_balanced.csv'
    
    try:
        df = preprocessor.load_data(data_path)
        print(f"  ✓ Loaded {len(df)} records")
    except FileNotFoundError:
        print(f"  ✗ Error: Dataset file not found at {data_path}")
        print(f"  Please ensure the dataset is available in the DATASET directory.")
        return
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        return
    
    try:
        X, y = preprocessor.preprocess_features(df)
        X_normalized = preprocessor.normalize_features(X)
        print(f"  ✓ Preprocessed features: {X_normalized.shape}")
        print(f"  ✓ Disaster types detected: {len(set(y))}")
    except Exception as e:
        print(f"  ✗ Error preprocessing data: {e}")
        return
    
    # Step 2: Train disaster classifier
    print("\n[2/5] Training disaster classification model...")
    classifier = DisasterClassifier(n_estimators=50, max_depth=10)
    metrics = classifier.train(X_normalized, y, test_size=0.2)
    print(f"  ✓ Training accuracy: {metrics['train_accuracy']:.2%}")
    print(f"  ✓ Test accuracy: {metrics['test_accuracy']:.2%}")
    
    # Display feature importance
    feature_names = ['Urgency', 'Temperature', 'Gas', 'Humidity', 'Flood Level']
    importances = classifier.get_feature_importance()
    print("\n  Feature Importance:")
    for name, importance in zip(feature_names, importances):
        print(f"    {name:15s}: {'█' * int(importance * 50)} {importance:.3f}")
    
    # Step 3: Create swarm network
    print("\n[3/5] Initializing swarm network...")
    coordinator = SwarmCoordinator()
    
    # Create edge nodes at different locations
    locations = [
        ("edge_node_1", (37.7749, -122.4194)),  # San Francisco
        ("edge_node_2", (34.0522, -118.2437)),  # Los Angeles
        ("edge_node_3", (40.7128, -74.0060)),   # New York
    ]
    
    for node_id, location in locations:
        node = EdgeNode(node_id, location)
        coordinator.register_node(node)
        print(f"  ✓ Registered {node_id} at {location}")
    
    # Broadcast model to all nodes
    coordinator.broadcast_model(classifier)
    print(f"  ✓ Model distributed to {len(coordinator.nodes)} edge nodes")
    
    # Step 4: Simulate sensor readings
    print("\n[4/5] Simulating disaster sensor readings...")
    
    # Generate test sensor readings
    test_readings = [
        {
            'urgency_level': 'High',
            'temperature_c': 45.0,
            'gas_ppm': 150,
            'humidity_percent': 15,
            'flood_sensor_level_cm': 0,
            'description': 'Potential wildfire - High temp, high gas'
        },
        {
            'urgency_level': 'High',
            'temperature_c': 25.0,
            'gas_ppm': 50,
            'humidity_percent': 85,
            'flood_sensor_level_cm': 120,
            'description': 'Potential flood - High water level'
        },
        {
            'urgency_level': 'Medium',
            'temperature_c': 30.0,
            'gas_ppm': 80,
            'humidity_percent': 60,
            'flood_sensor_level_cm': 0,
            'description': 'Building collapse scenario'
        },
    ]
    
    for i, reading in enumerate(test_readings, 1):
        desc = reading.pop('description')
        print(f"\n  Sensor Reading #{i}: {desc}")
        print(f"    Temperature: {reading['temperature_c']}°C")
        print(f"    Gas Level: {reading['gas_ppm']} ppm")
        print(f"    Humidity: {reading['humidity_percent']}%")
        print(f"    Flood Level: {reading['flood_sensor_level_cm']} cm")
    
    # Step 5: Process with swarm intelligence
    print("\n[5/5] Processing with swarm intelligence...")
    result = coordinator.process_distributed(test_readings)
    
    if 'error' not in result:
        pred_idx = result['swarm_prediction']
        disaster_type = preprocessor.get_label_name(pred_idx)
        
        print(f"\n  🎯 Swarm Consensus Result:")
        print(f"    Disaster Type: {disaster_type}")
        print(f"    Consensus Strength: {result['consensus_strength']:.1%}")
        print(f"    Average Confidence: {result['average_confidence']:.1%}")
        print(f"    Nodes Participating: {result['voting_nodes']}/{result['total_nodes']}")
        
        if result['high_consensus']:
            print(f"    ✓ HIGH CONSENSUS - Reliable prediction")
        else:
            print(f"    ⚠ LOW CONSENSUS - Results uncertain")
            
        # Show individual node predictions
        print(f"\n  Individual Node Predictions:")
        for i, pred in enumerate(result['individual_predictions'], 1):
            node_disaster = preprocessor.get_label_name(pred['prediction'])
            print(f"    {pred['node_id']}: {node_disaster} (confidence: {pred['confidence']:.1%})")
    else:
        print(f"  ✗ Error: {result['error']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    print("\nKey Capabilities Demonstrated:")
    print("  ✓ Multimodal sensor data processing")
    print("  ✓ Edge-based disaster classification")
    print("  ✓ Distributed swarm intelligence")
    print("  ✓ Consensus-based decision making")
    print("  ✓ Fault-tolerant disaster response")
    print("\nThis system can be deployed on IoT edge devices for")
    print("real-time disaster detection and response coordination.")
    print()


if __name__ == "__main__":
    main()
