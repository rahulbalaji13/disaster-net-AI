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
        # Encode urgency level with default for unknown values
        df['urgency_encoded'] = df['urgency_level'].map(self.urgency_mapping)
        if df['urgency_encoded'].isna().any():
            unknown_urgencies = df[df['urgency_encoded'].isna()]['urgency_level'].unique()
            raise ValueError(f"Unknown urgency levels found: {unknown_urgencies}. "
                           f"Expected one of: {list(self.urgency_mapping.keys())}")
        
        # Encode drone image labels
        unique_labels = df['drone_image_label'].unique()
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        df['label_encoded'] = df['drone_image_label'].map(self.label_mapping)
        if df['label_encoded'].isna().any():
            raise ValueError("Error encoding drone image labels")
        
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
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_range = X_max - X_min
        
        # Handle constant features (where min == max)
        constant_features = X_range == 0
        if constant_features.any():
            # For constant features, just return 0.5 (middle of normalized range)
            X_range[constant_features] = 1.0
        
        return (X - X_min) / X_range
    
    def get_label_name(self, label_idx: int) -> str:
        """Get disaster label name from encoded index."""
        reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        return reverse_mapping.get(label_idx, "Unknown")
