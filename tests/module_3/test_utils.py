import numpy as np
import pandas as pd
import pytest

from utils import FEATURE_COLUMNS, LABEL_COLUMN, evaluate_model, filter_data, split_data


# ---- Fixtures ----

@pytest.fixture
def sample_df():
    """Small synthetic dataframe that mimics the real data structure."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "order_id": np.repeat(np.arange(200), 5),
        "order_date": pd.date_range("2020-10-01", periods=n, freq="h"),
        "outcome": np.random.randint(0, 2, n).astype("float32"),
        "ordered_before": np.random.randint(0, 2, n).astype("float32"),
        "global_popularity": np.random.uniform(0, 1, n).astype("float32"),
        "abandoned_before": np.random.randint(0, 2, n).astype("float32"),
    })


# ---- filter_data ----

def test_filter_data_removes_small_orders(sample_df):
    df_filtered = filter_data(sample_df, min_products=5)
    order_sizes = df_filtered.groupby("order_id").outcome.sum()
    assert (order_sizes >= 5).all()


def test_filter_data_reduces_rows(sample_df):
    df_filtered = filter_data(sample_df, min_products=5)
    assert len(df_filtered) <= len(sample_df)


def test_filter_data_keeps_columns(sample_df):
    df_filtered = filter_data(sample_df, min_products=5)
    assert set(sample_df.columns) == set(df_filtered.columns)


def test_filter_data_empty_result_with_high_threshold(sample_df):
    df_filtered = filter_data(sample_df, min_products=9999)
    assert len(df_filtered) == 0


# ---- split_data ----

def test_split_data_no_leakage(sample_df):
    df_train, df_val, df_test = split_data(sample_df)
    assert df_train["order_date"].max() < df_val["order_date"].min()
    assert df_val["order_date"].max() < df_test["order_date"].min()


def test_split_data_covers_all_rows(sample_df):
    df_train, df_val, df_test = split_data(sample_df)
    assert len(df_train) + len(df_val) + len(df_test) == len(sample_df)


def test_split_data_non_empty(sample_df):
    df_train, df_val, df_test = split_data(sample_df)
    assert len(df_train) > 0
    assert len(df_val) > 0
    assert len(df_test) > 0


def test_split_data_keeps_columns(sample_df):
    df_train, df_val, df_test = split_data(sample_df)
    for df in [df_train, df_val, df_test]:
        assert set(df.columns) == set(sample_df.columns)


# ---- evaluate_model ----

class DummyModel:
    """Fake model that always predicts 0.5."""
    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.5), np.full(len(X), 0.5)])


def test_evaluate_model_returns_correct_keys(sample_df):
    metrics = evaluate_model(DummyModel(), sample_df[FEATURE_COLUMNS], sample_df[LABEL_COLUMN], "Test")
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics


def test_evaluate_model_auc_in_valid_range(sample_df):
    metrics = evaluate_model(DummyModel(), sample_df[FEATURE_COLUMNS], sample_df[LABEL_COLUMN], "Test")
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["pr_auc"] <= 1