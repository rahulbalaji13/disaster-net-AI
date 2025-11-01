"""
Data preprocessing utilities for disaster response system.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict


class DataPreprocessor:
    """Preprocessor for multimodal disaster dataset."""
    
    def __init__(self):
        self.urgency_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        self.label_mapping = {}
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load dataset from CSV file."""
        return pd.read_csv(filepath)
    
    def preprocess_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features for model training.
        
        Args:
            df: DataFrame containing the disaster data
            
        Returns:
            Tuple of (features, labels)
        """
        # Encode urgency level
        df['urgency_encoded'] = df['urgency_level'].map(self.urgency_mapping)
        
        # Encode drone image labels
        unique_labels = df['drone_image_label'].unique()
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        df['label_encoded'] = df['drone_image_label'].map(self.label_mapping)
        
        # Select sensor features
        feature_columns = [
            'urgency_encoded', 
            'temperature_c', 
            'gas_ppm', 
            'humidity_percent',
            'flood_sensor_level_cm'
        ]
        
        X = df[feature_columns].values
        y = df['label_encoded'].values
        
        return X, y
    
    def normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features to 0-1 range."""
        return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    
    def get_label_name(self, label_idx: int) -> str:
        """Get disaster label name from encoded index."""
        reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        return reverse_mapping.get(label_idx, "Unknown")
