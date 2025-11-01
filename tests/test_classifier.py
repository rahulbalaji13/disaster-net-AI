"""Unit tests for disaster classifier."""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.classifier import DisasterClassifier


class TestDisasterClassifier(unittest.TestCase):
    """Test cases for DisasterClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = DisasterClassifier(n_estimators=10, max_depth=5)
        
        # Create sample training data
        np.random.seed(42)
        self.X_train = np.random.rand(100, 5)
        self.y_train = np.random.randint(0, 3, 100)
    
    def test_initialization(self):
        """Test classifier initialization."""
        self.assertIsNotNone(self.classifier.model)
        self.assertFalse(self.classifier.is_trained)
    
    def test_train(self):
        """Test model training."""
        metrics = self.classifier.train(self.X_train, self.y_train)
        
        # Check that training completed
        self.assertTrue(self.classifier.is_trained)
        
        # Check metrics
        self.assertIn('train_accuracy', metrics)
        self.assertIn('test_accuracy', metrics)
        self.assertGreater(metrics['train_accuracy'], 0)
        self.assertGreater(metrics['test_accuracy'], 0)
    
    def test_predict_before_training(self):
        """Test that prediction fails before training."""
        X_test = np.random.rand(5, 5)
        
        with self.assertRaises(ValueError):
            self.classifier.predict(X_test)
    
    def test_predict_after_training(self):
        """Test prediction after training."""
        self.classifier.train(self.X_train, self.y_train)
        
        X_test = np.random.rand(5, 5)
        predictions = self.classifier.predict(X_test)
        
        # Check predictions
        self.assertEqual(len(predictions), 5)
        self.assertTrue(np.all(predictions >= 0))
    
    def test_predict_proba(self):
        """Test probability prediction."""
        self.classifier.train(self.X_train, self.y_train)
        
        X_test = np.random.rand(5, 5)
        probabilities = self.classifier.predict_proba(X_test)
        
        # Check shape
        self.assertEqual(probabilities.shape[0], 5)
        
        # Check that probabilities sum to 1
        prob_sums = probabilities.sum(axis=1)
        np.testing.assert_array_almost_equal(prob_sums, np.ones(5))
    
    def test_feature_importance(self):
        """Test feature importance retrieval."""
        self.classifier.train(self.X_train, self.y_train)
        
        importances = self.classifier.get_feature_importance()
        
        # Check shape and values
        self.assertEqual(len(importances), 5)
        self.assertTrue(np.all(importances >= 0))
        self.assertTrue(np.all(importances <= 1))
        
        # Check that importances sum to approximately 1
        self.assertAlmostEqual(importances.sum(), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
