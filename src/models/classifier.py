"""
Disaster classification model using edge-compatible algorithms.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict


class DisasterClassifier:
    """
    Lightweight disaster classification model suitable for edge deployment.
    Uses Random Forest for robustness and low computational requirements.
    """
    
    def __init__(self, n_estimators: int = 50, max_depth: int = 10):
        """
        Initialize the classifier.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict:
        """
        Train the classifier on disaster data.
        
        Args:
            X: Feature matrix (sensor readings)
            y: Labels (disaster types)
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with training metrics
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict disaster type from sensor readings.
        
        Args:
            X: Feature matrix (sensor readings)
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict disaster type probabilities.
        
        Args:
            X: Feature matrix (sensor readings)
            
        Returns:
            Probability matrix
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.feature_importances_
