"""
Unified Classifier Module

This module provides a single classifier class that can handle multiple classifier types:
- SVM (Support Vector Machine)
- Neural Networks (MLPClassifier)
- Decision Trees (Random Forest, XGBoost)
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from typing import Union, Dict, Any, Optional, Tuple
import pickle
import os


class UnifiedClassifier:
    """
    A unified classifier class that can handle multiple classifier types.
    """
    
    def __init__(self, 
                 classifier_type: str = "svm",
                 **kwargs):
        """
        Initialize the unified classifier.
        
        Args:
            classifier_type: Type of classifier ('svm', 'neural_network', 'random_forest', 'xgboost')
            **kwargs: Additional parameters for the specific classifier
        """
        self.classifier_type = classifier_type.lower()
        self.classifier = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        
        # Initialize the classifier based on type
        self._initialize_classifier(**kwargs)
    
    def _initialize_classifier(self, **kwargs):
        """Initialize the classifier based on the specified type."""
        if self.classifier_type == "svm":
            self.classifier = SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                C=kwargs.get('C', 1.0),
                gamma=kwargs.get('gamma', 'scale'),
                random_state=kwargs.get('random_state', 42),
                probability=kwargs.get('probability', True)
            )
        
        elif self.classifier_type == "neural_network":
            self.classifier = MLPClassifier(
                hidden_layer_sizes=kwargs.get('hidden_layer_sizes', (100, 50)),
                activation=kwargs.get('activation', 'relu'),
                solver=kwargs.get('solver', 'adam'),
                alpha=kwargs.get('alpha', 0.0001),
                learning_rate=kwargs.get('learning_rate', 'adaptive'),
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42)
            )
        
        elif self.classifier_type == "random_forest":
            self.classifier = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                min_samples_split=kwargs.get('min_samples_split', 2),
                min_samples_leaf=kwargs.get('min_samples_leaf', 1),
                random_state=kwargs.get('random_state', 42)
            )
        
        elif self.classifier_type == "xgboost":
            self.classifier = xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=kwargs.get('random_state', 42)
            )
        
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}. "
                           f"Supported types: 'svm', 'neural_network', 'random_forest', 'xgboost'")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            scale_features: bool = True) -> 'UnifiedClassifier':
        """
        Fit the classifier on the training data.
        
        Args:
            X: Feature matrix
            y: Target labels
            scale_features: Whether to scale features (recommended for SVM and Neural Networks)
            
        Returns:
            Self for method chaining
        """
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Scale features if requested
        if scale_features and self.classifier_type in ['svm', 'neural_network']:
            X = self.scaler.fit_transform(X)
        
        # Fit the classifier
        self.classifier.fit(X, y)
        self.is_trained = True
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before making predictions")
        
        X = np.array(X)
        
        # Scale features if scaler was used during training
        if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
            X = self.scaler.transform(X)
        
        return self.classifier.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before making predictions")
        
        X = np.array(X)
        
        # Scale features if scaler was used during training
        if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
            X = self.scaler.transform(X)
        
        return self.classifier.predict_proba(X)
    
    def evaluate(self, X: Union[np.ndarray, pd.DataFrame], 
                y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Evaluate the classifier on test data.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before evaluation")
        
        y_pred = self.predict(X)
        y_true = np.array(y)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Get detailed classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Get confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    
    def cross_validate(self, X: Union[np.ndarray, pd.DataFrame], 
                      y: Union[np.ndarray, pd.Series],
                      cv: int = 5,
                      scale_features: bool = True) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of cross-validation folds
            scale_features: Whether to scale features
            
        Returns:
            Dictionary with cross-validation results
        """
        X = np.array(X)
        y = np.array(y)
        
        # Scale features if requested
        if scale_features and self.classifier_type in ['svm', 'neural_network']:
            X = self.scaler.fit_transform(X)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.classifier, X, y, cv=cv, scoring='accuracy')
        
        return {
            'cv_scores': cv_scores,
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'min_accuracy': cv_scores.min(),
            'max_accuracy': cv_scores.max()
        }
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores (for tree-based models).
        
        Returns:
            Feature importance scores or None if not available
        """
        if not self.is_trained:
            return None
        
        if hasattr(self.classifier, 'feature_importances_'):
            return self.classifier.feature_importances_
        else:
            return None
    
    def get_feature_importance_df(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance as a DataFrame with feature names.
        
        Returns:
            DataFrame with feature importance or None if not available
        """
        importance = self.get_feature_importance()
        if importance is None:
            return None
        
        if self.feature_names:
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            return pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importance))],
                'importance': importance
            }).sort_values('importance', ascending=False)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'classifier_type': self.classifier_type,
            'classifier': self.classifier,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> 'UnifiedClassifier':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Self for method chaining
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier_type = model_data['classifier_type']
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'classifier_type': self.classifier_type,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }
        
        if self.is_trained and hasattr(self.classifier, 'n_features_in_'):
            info['n_features'] = self.classifier.n_features_in_
        
        if self.is_trained and hasattr(self.classifier, 'classes_'):
            info['n_classes'] = len(self.classifier.classes_)
            info['classes'] = self.classifier.classes_.tolist()
        
        return info


def train_test_classifier(X: Union[np.ndarray, pd.DataFrame],
                         y: Union[np.ndarray, pd.Series],
                         classifier_type: str = "svm",
                         test_size: float = 0.2,
                         random_state: int = 42,
                         scale_features: bool = True,
                         **classifier_kwargs) -> Tuple[UnifiedClassifier, Dict[str, Any]]:
    """
    Train and test a classifier with automatic train-test split.
    
    Args:
        X: Feature matrix
        y: Target labels
        classifier_type: Type of classifier
        test_size: Proportion of data for testing
        random_state: Random seed
        scale_features: Whether to scale features
        **classifier_kwargs: Additional classifier parameters
        
    Returns:
        Tuple of (trained_classifier, evaluation_results)
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize and train classifier
    classifier = UnifiedClassifier(classifier_type=classifier_type, **classifier_kwargs)
    classifier.fit(X_train, y_train, scale_features=scale_features)
    
    # Evaluate on test set
    evaluation_results = classifier.evaluate(X_test, y_test)
    
    return classifier, evaluation_results


if __name__ == "__main__":
    pass
