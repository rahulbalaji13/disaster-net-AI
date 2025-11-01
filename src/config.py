"""Configuration settings for DisasterNet SwarmAI."""

# Model Configuration
MODEL_CONFIG = {
    'n_estimators': 50,
    'max_depth': 10,
    'random_state': 42,
}

# Swarm Configuration
SWARM_CONFIG = {
    'consensus_threshold': 0.7,
    'max_nodes': 100,
}

# Edge Node Configuration
EDGE_CONFIG = {
    'max_cache_size': 100,
    'processing_timeout': 5.0,  # seconds
}

# Data Configuration
DATA_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
}

# Feature names
FEATURE_NAMES = [
    'urgency_level',
    'temperature_c',
    'gas_ppm',
    'humidity_percent',
    'flood_sensor_level_cm'
]

# Disaster types mapping
DISASTER_TYPES = {
    'Collapsed_Building': 'Building Collapse',
    'Fire': 'Wildfire',
    'Flood': 'Flood',
    'Earthquake': 'Earthquake',
}
