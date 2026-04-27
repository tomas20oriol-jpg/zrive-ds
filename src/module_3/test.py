import logging
import pickle
from pathlib import Path

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


def load_model(path: Path):
    logger.info(f"Loading model from {path}...")
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded")
    return model


def main():
    PROJECT_ROOT = find_project_root()
    data_path = PROJECT_ROOT / "data/raw"
    model_path = PROJECT_ROOT / "models/model.pkl"

    # Load and prepare data
    df = load_data(data_path)
    df_filtered = filter_data(df)
    del df  # free memory immediately after filtering
    _, _, df_test = split_data(df_filtered)
    del df_filtered  # free memory after splitting

    X_test = df_test[FEATURE_COLUMNS]
    y_test = df_test[LABEL_COLUMN]
    del df_test  # free memory before inference

    # Load model and evaluate
    model = load_model(model_path)
    metrics = evaluate_model(model, X_test, y_test, "Test")

    logger.info(f"Final Test ROC AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Final Test PR AUC:  {metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    main()