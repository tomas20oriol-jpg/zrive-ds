import warnings
import pandas as pd
from typing import Tuple, Dict
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier


class PushModel(BaseEstimator):
    # Features used for training and inference
    MODEL_COLUMNS = [
        "ordered_before",
        "abandoned_before",
        "global_popularity",
    ]

    TARGET_COLUMN = "outcome"

    def __init__(
            self,
            classifier_parametrisation: Dict,
            calibration_parametrisation: Dict,
            prediction_threshold: float) -> None:
        # Wrap GBT with calibration to produce reliable probabilities
        self.clf = CalibratedClassifierCV(
            estimator=GradientBoostingClassifier(**classifier_parametrisation),
            **calibration_parametrisation
        )
        self.prediction_threshold = prediction_threshold

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Select only the model's expected feature columns
        return df.loc[:, self.MODEL_COLUMNS]

    def _extract_labels(self, df: pd.DataFrame) -> pd.Series:
        # Extract the binary target column
        return df.loc[:, self.TARGET_COLUMN]

    def _feature_label_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        # Convenience method to split a dataframe into features and labels
        return self._extract_features(df), self._extract_labels(df)

    def fit(self, df: pd.DataFrame) -> None:
        # Extract features and labels then fit the calibrated classifier
        features, labels = self._feature_label_split(df)
        self.clf.fit(features, labels)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        # Apply threshold to calibrated probabilities to generate binary predictions
        probs = self.predict_proba(df)
        return (probs >= self.prediction_threshold).astype(int)

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        # Return calibrated probabilities for the positive class
        features = self._extract_features(df)
        predictions = pd.Series(
            self.clf.predict_proba(features)[:, 1],
            name="predictions"
        )
        # Preserve original dataframe index for downstream alignment
        if hasattr(features, "index"):
            predictions.index = features.index
        return predictions