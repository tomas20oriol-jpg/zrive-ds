import joblib
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

from .push_model import PushModel
from .utils import load_training_feature_frame

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Default folder where trained models are persisted
DEFAULT_MODEL_FOLDER_PATH = Path(__file__).parent.parent.parent / "models"

# Best hyperparameters found during notebook experimentation
DEFAULT_CLASSIFIER_PARAMS: Dict = {
    "learning_rate": 0.1,
    "max_depth": 5,
    "n_estimators": 100,
}

# Calibration using prefit since we calibrate on a held-out validation set
DEFAULT_CALIBRATION_PARAMS: Dict = {
    "cv": "prefit",
    "method": "isotonic",
}

# Minimum probability of a user opening the notification
DEFAULT_PREDICTION_THRESHOLD: float = 0.05


def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train, validation and test sets respecting temporal order."""
    df = df.sort_values("order_date").reset_index(drop=True)
    cumulative_orders = df.groupby("order_date").size().cumsum()
    total_orders = len(df)

    # 70% train / 20% validation / 10% test by cumulative order volume
    train_cutoff = cumulative_orders[cumulative_orders <= total_orders * 0.70].index.max()
    val_cutoff   = cumulative_orders[cumulative_orders <= total_orders * 0.90].index.max()

    df_train = df[df["order_date"] <= train_cutoff]
    df_val   = df[(df["order_date"] > train_cutoff) & (df["order_date"] <= val_cutoff)]
    df_test  = df[df["order_date"] > val_cutoff]

    logger.info(f"Train: {len(df_train)} rows | Val: {len(df_val)} rows | Test: {len(df_test)} rows")
    return df_train, df_val, df_test


def train(
    classifier_params: Dict = DEFAULT_CLASSIFIER_PARAMS,
    calibration_params: Dict = DEFAULT_CALIBRATION_PARAMS,
    prediction_threshold: float = DEFAULT_PREDICTION_THRESHOLD,
) -> PushModel:
    """Load data, train and calibrate the PushModel, then persist it to disk."""

    # Load and preprocess the feature frame
    logger.info("Loading feature frame...")
    df = load_training_feature_frame()

    # Temporal split — train on past, calibrate on validation, evaluate on test
    df_train, df_val, _ = temporal_split(df)

    # Initialise model with given hyperparameters
    model = PushModel(
        classifier_parametrisation=classifier_params,
        calibration_parametrisation=calibration_params,
        prediction_threshold=prediction_threshold,
    )

    # Fit base GBT on training set
    logger.info("Fitting model on training set...")
    model.clf.estimator.fit(
        df_train[PushModel.MODEL_COLUMNS],
        df_train[PushModel.TARGET_COLUMN],
    )

    # Calibrate probabilities on held-out validation set
    logger.info("Calibrating model on validation set...")
    model.clf.fit(
        df_val[PushModel.MODEL_COLUMNS],
        df_val[PushModel.TARGET_COLUMN],
    )

    # Persist trained model to disk with today's date in the filename
    DEFAULT_MODEL_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    today_str = datetime.today().strftime("%Y-%m-%d")
    model_path = DEFAULT_MODEL_FOLDER_PATH / f"push_{today_str}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    return model


if __name__ == "__main__":
    train()