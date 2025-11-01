"""Unit tests for data preprocessing utilities."""
import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.preprocessor import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'record_id': [1, 2, 3, 4, 5],
            'sos_message': ['Help!', 'Fire!', 'Flood!', 'Earthquake!', 'Trapped!'],
            'urgency_level': ['Low', 'High', 'Medium', 'High', 'Low'],
            'temperature_c': [25.0, 45.0, 20.0, 30.0, 22.0],
            'gas_ppm': [50, 150, 60, 80, 45],
            'humidity_percent': [60, 20, 85, 55, 70],
            'flood_sensor_level_cm': [0, 0, 120, 0, 0],
            'drone_image_label': ['Fire', 'Fire', 'Flood', 'Earthquake', 'Collapsed_Building']
        })
    
    def test_urgency_mapping(self):
        """Test urgency level encoding."""
        self.assertEqual(self.preprocessor.urgency_mapping['Low'], 0)
        self.assertEqual(self.preprocessor.urgency_mapping['Medium'], 1)
        self.assertEqual(self.preprocessor.urgency_mapping['High'], 2)
    
    def test_preprocess_features(self):
        """Test feature preprocessing."""
        X, y = self.preprocessor.preprocess_features(self.sample_data)
        
        # Check shapes
        self.assertEqual(X.shape[0], 5)  # 5 samples
        self.assertEqual(X.shape[1], 5)  # 5 features
        self.assertEqual(y.shape[0], 5)  # 5 labels
        
        # Check types
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
    
    def test_normalize_features(self):
        """Test feature normalization."""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        X_norm = self.preprocessor.normalize_features(X)
        
        # Check that values are normalized
        self.assertTrue(np.all(X_norm >= 0))
        self.assertTrue(np.all(X_norm <= 1))
        
        # Check shape preservation
        self.assertEqual(X_norm.shape, X.shape)
    
    def test_label_mapping(self):
        """Test label encoding."""
        X, y = self.preprocessor.preprocess_features(self.sample_data)
        
        # Check that labels are integers
        self.assertTrue(np.all(y >= 0))
        self.assertTrue(np.issubdtype(y.dtype, np.integer))
        
        # Check label mapping was created
        self.assertGreater(len(self.preprocessor.label_mapping), 0)
    
    def test_get_label_name(self):
        """Test label name retrieval."""
        X, y = self.preprocessor.preprocess_features(self.sample_data)
        
        # Get a label name
        label_idx = y[0]
        label_name = self.preprocessor.get_label_name(label_idx)
        
        self.assertIsInstance(label_name, str)
        self.assertIn(label_name, self.sample_data['drone_image_label'].values)


if __name__ == '__main__':
    unittest.main()
