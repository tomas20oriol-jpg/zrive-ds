import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
FEATURE_COLUMNS = ["ordered_before", "global_popularity", "abandoned_before"]
LABEL_COLUMN = "outcome"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN, "order_id", "order_date"]
DTYPES = {
    "order_id": "int32",
    "outcome": "float32",
    "ordered_before": "float32",
    "global_popularity": "float32",
    "abandoned_before": "float32",
}


def find_project_root() -> Path:
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("pyproject.toml not found in project root")


def load_data(data_path: Path) -> pd.DataFrame:
    logger.info("Loading data...")
    df = pd.read_csv(
        f"{data_path}/feature_frame.csv",
        usecols=REQUIRED_COLUMNS,
        dtype=DTYPES,
        parse_dates=["order_date"],
    )
    logger.info(f"Loaded {len(df)} rows — memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


def filter_data(df: pd.DataFrame, min_products: int = 5) -> pd.DataFrame:
    logger.info(f"Filtering orders with at least {min_products} products...")
    order_size = df.groupby("order_id").outcome.sum()
    orders_of_min_size = order_size[order_size >= min_products].index
    df_filtered = df[df["order_id"].isin(orders_of_min_size)].copy()
    logger.info(f"Filtered to {len(df_filtered)} rows ({df_filtered['order_id'].nunique()} orders)")
    return df_filtered


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Splitting data temporally...")
    daily_orders = df.groupby("order_date").order_id.nunique()
    cum_sum = daily_orders.cumsum() / daily_orders.sum()

    cutoff_train = cum_sum[cum_sum <= 0.70].index[-1]
    cutoff_val = cum_sum[cum_sum <= 0.90].index[-1]

    df_train = df[df["order_date"] <= cutoff_train]
    df_val = df[(df["order_date"] > cutoff_train) & (df["order_date"] <= cutoff_val)]
    df_test = df[df["order_date"] > cutoff_val]

    logger.info(f"Train: {df_train['order_date'].min().date()} → {df_train['order_date'].max().date()} ({len(df_train)} rows)")
    logger.info(f"Val:   {df_val['order_date'].min().date()} → {df_val['order_date'].max().date()} ({len(df_val)} rows)")
    logger.info(f"Test:  {df_test['order_date'].min().date()} → {df_test['order_date'].max().date()} ({len(df_test)} rows)")

    return df_train, df_val, df_test


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, split_name: str) -> dict:
    y_pred = model.predict_proba(X)[:, 1]
    roc_auc = roc_auc_score(y, y_pred)
    precision, recall, _ = precision_recall_curve(y, y_pred)
    pr_auc = auc(recall, precision)
    logger.info(f"{split_name} — ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f}")
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def plot_metrics(
    model_name: str,
    y_pred: pd.Series,
    y_test: pd.Series,
    target_precision: float = 0.05,
    figure: Tuple = None,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.figure

    if figure is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        show = True
    else:
        fig, axes = figure
        show = False

    if not axes[0].lines:
        axes[0].plot([0, 1], [0, 1], "k--", label="Random")
    if not axes[1].lines:
        axes[1].axhline(
            y=target_precision,
            color="r",
            linestyle="--",
            label=f"Target precision ({target_precision})",
        )

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    axes[0].plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)
    axes[1].plot(recall, precision, label=f"{model_name} (AUC = {pr_auc:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    plt.tight_layout()

    if show:
        plt.show()