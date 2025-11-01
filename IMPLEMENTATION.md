# DisasterNet SwarmAI - Implementation Guide

## Overview

This implementation provides a complete swarm-enabled edge intelligence system for disaster response using IoT devices. The system combines multimodal sensor data with distributed AI processing to detect and classify disasters in real-time.

## Architecture

### Core Components

1. **Data Preprocessing** (`src/utils/preprocessor.py`)
   - Loads and preprocesses multimodal disaster datasets
   - Normalizes sensor readings
   - Encodes categorical variables

2. **Disaster Classifier** (`src/models/classifier.py`)
   - Lightweight Random Forest model for edge deployment
   - Trained on sensor data (temperature, gas, humidity, flood levels)
   - Provides probability estimates for disaster types

3. **Edge Node** (`src/edge/edge_node.py`)
   - Represents individual IoT edge devices
   - Performs local AI inference to reduce latency
   - Caches sensor readings for analysis

4. **Swarm Coordinator** (`src/swarm/coordinator.py`)
   - Coordinates multiple edge nodes
   - Implements consensus-based decision making
   - Distributes models and aggregates predictions

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Usage

### Running the Demo

```bash
python demo.py
```

This will:
1. Load the disaster dataset
2. Train a classification model
3. Create a swarm network of edge nodes
4. Simulate sensor readings
5. Demonstrate distributed disaster detection

### Using the Components

```python
from models.classifier import DisasterClassifier
from utils.preprocessor import DataPreprocessor
from edge.edge_node import EdgeNode
from swarm.coordinator import SwarmCoordinator

# Load and preprocess data
preprocessor = DataPreprocessor()
df = preprocessor.load_data('DATASET/disaster_multimodal_dataset_10000_balanced.csv')
X, y = preprocessor.preprocess_features(df)
X_normalized = preprocessor.normalize_features(X)

# Train classifier
classifier = DisasterClassifier()
metrics = classifier.train(X_normalized, y)

# Create swarm network
coordinator = SwarmCoordinator()
node1 = EdgeNode("node_1", (37.7749, -122.4194))
coordinator.register_node(node1)
coordinator.broadcast_model(classifier)

# Process sensor data
sensor_reading = {
    'urgency_level': 'High',
    'temperature_c': 45.0,
    'gas_ppm': 150,
    'humidity_percent': 20,
    'flood_sensor_level_cm': 0
}
result = node1.process_sensor_data(sensor_reading)
```

## Testing

Run all tests:
```bash
python -m unittest discover tests
```

Run specific test:
```bash
python -m unittest tests.test_classifier
```

## Dataset

The system uses the disaster multimodal dataset with the following features:
- **sos_message**: Emergency message text
- **urgency_level**: Low, Medium, or High
- **temperature_c**: Temperature in Celsius
- **gas_ppm**: Gas concentration in parts per million
- **humidity_percent**: Humidity percentage
- **flood_sensor_level_cm**: Water level in centimeters
- **drone_image_label**: Visual disaster classification

## Features

### Edge Intelligence
- Local AI processing reduces latency and bandwidth
- Fault-tolerant design continues operating with connectivity loss
- Lightweight models suitable for resource-constrained IoT devices

### Swarm Coordination
- Distributed decision making across multiple nodes
- Consensus-based predictions improve accuracy
- Scalable architecture supports growing networks

### Disaster Detection
Supports multiple disaster types:
- Forest fires / Wildfires
- Floods
- Earthquakes
- Building collapses
- And more based on sensor signatures

## Configuration

Edit `src/config.py` to customize:
- Model parameters (tree depth, ensemble size)
- Swarm settings (consensus threshold, max nodes)
- Edge node settings (cache size, timeout)

## Project Structure

```
disasterNet-swarmAI/
├── src/
│   ├── models/         # ML models for disaster classification
│   ├── utils/          # Data preprocessing utilities
│   ├── edge/           # Edge computing components
│   ├── swarm/          # Swarm coordination logic
│   └── config.py       # Configuration settings
├── tests/              # Unit tests
├── DATASET/            # Disaster datasets
├── demo.py             # Demonstration script
├── requirements.txt    # Python dependencies
└── setup.py           # Package setup
```

## Contributing

This is an open-source disaster response system. Contributions are welcome!

## License

See LICENSE file for details.

## Author

Made with ❤️ by Rahul Balaji
