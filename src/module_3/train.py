import logging
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    evaluate_model,
    filter_data,
    find_project_root,
    load_data,
    split_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_model(X_train, y_train, C: float = 4.64e-6):
    logger.info(f"Training Ridge Logistic Regression with C={C}...")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l2", C=C, max_iter=1000, solver="saga")
    )
    model.fit(X_train, y_train)
    logger.info("Training complete")
    return model


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")


def main():
    PROJECT_ROOT = find_project_root()
    data_path = PROJECT_ROOT / "data/raw"
    model_path = PROJECT_ROOT / "models/model.pkl"

    # Load and prepare data
    df = load_data(data_path)
    df_filtered = filter_data(df)
    del df  # free memory immediately after filtering
    df_train, df_val, df_test = split_data(df_filtered)
    del df_filtered  # free memory after splitting

    X_train, y_train = df_train[FEATURE_COLUMNS], df_train[LABEL_COLUMN]
    X_val, y_val = df_val[FEATURE_COLUMNS], df_val[LABEL_COLUMN]
    X_test, y_test = df_test[FEATURE_COLUMNS], df_test[LABEL_COLUMN]
    del df_train, df_val, df_test  # free memory before training

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    evaluate_model(model, X_train, y_train, "Train")
    evaluate_model(model, X_val, y_val, "Validation")
    evaluate_model(model, X_test, y_test, "Test")

    # Save
    save_model(model, model_path)


if __name__ == "__main__":
    main()