from pathlib import Path
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.level = logging.INFO

def load_raw_dataset() -> pd.DataFrame:
    loading_file = (
        Path(__file__)
        .parent
        .parent
        .parent.resolve()
        / "data"
        / "raw"
        / "feature_frame_20210304.csv"
    )
    logger.info(f"Loading dataset from {loading_file}")
    return pd.read_csv(loading_file)

def push_relevant_orders(df: pd.DataFrame, min_products: int = 5) -> pd.DataFrame:
    logger.info(f"Filtering orders with at least {min_products} products...")
    order_size = df.groupby("order_id").size()
    orders_of_min_size = order_size[order_size >= min_products].index
    return df.loc[lambda x: x.order_id.isin(orders_of_min_size), :]

def load_training_feature_frame() -> pd.DataFrame:
    df = load_raw_dataset().pipe(push_relevant_orders)
    feature_frame = build_feature_frame(df)
    return feature_frame

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building feature frame...")
    return df.assign(
        created_at=lambda x: pd.to_datetime(x.created_at),
        order_date=lambda x: pd.to_datetime(x.order_date).dt.date
    )