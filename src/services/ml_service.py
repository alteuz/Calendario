import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

class ModelExecutionError(Exception):
    """Exception raised when machine learning operations fail."""
    pass

class WeatherClassifierAdapter:
    """Adapter for the scikit-learn machine learning classifier."""

    def __init__(self) -> None:
        self._imputer: SimpleImputer = SimpleImputer(strategy='median')
        self._classifier: RandomForestClassifier = RandomForestClassifier(
            random_state=42, 
            class_weight='balanced'
        )
        self._is_trained: bool = False

    def train_classifier(self, feature_matrix: pd.DataFrame, target_series: pd.Series) -> None:
        """Fits the imputer and classifier model on the provided data."""
        try:
            imputed_features = self._imputer.fit_transform(feature_matrix)
            self._classifier.fit(imputed_features, target_series)
            self._is_trained = True
        except Exception as error:
            raise ModelExecutionError("Failed to train the weather classifier model.") from error

    def predict_weather_labels(self, feature_matrix: pd.DataFrame) -> np.ndarray:
        """Generates weather predictions for the input features."""
        if not self._is_trained:
            raise ModelExecutionError("Classifier model must be trained before predicting.")
        try:
            imputed_features = self._imputer.transform(feature_matrix)
            predictions: np.ndarray = self._classifier.predict(imputed_features)
            return predictions
        except Exception as error:
            raise ModelExecutionError("Error generating weather predictions.") from error

    @property
    def is_trained(self) -> bool:
        """Returns True if the classifier is trained and ready."""
        return self._is_trained
