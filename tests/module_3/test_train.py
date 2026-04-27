import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from train import train_model, save_model
from utils import FEATURE_COLUMNS, LABEL_COLUMN


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


@pytest.fixture
def trained_model(sample_df):
    X = sample_df[FEATURE_COLUMNS]
    y = sample_df[LABEL_COLUMN]
    return train_model(X, y)


# ---- train_model ----

def test_train_model_returns_pipeline(trained_model):
    assert isinstance(trained_model, Pipeline)


def test_train_model_can_predict(sample_df, trained_model):
    X = sample_df[FEATURE_COLUMNS]
    proba = trained_model.predict_proba(X)[:, 1]
    assert len(proba) == len(X)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_train_model_different_c_values_give_different_results(sample_df):
    X = sample_df[FEATURE_COLUMNS]
    y = sample_df[LABEL_COLUMN]
    model_1 = train_model(X, y, C=1e-6)
    model_2 = train_model(X, y, C=1.0)
    proba_1 = model_1.predict_proba(X)[:, 1]
    proba_2 = model_2.predict_proba(X)[:, 1]
    assert not np.allclose(proba_1, proba_2)


# ---- save_model / load_model ----

def test_save_and_load_model_returns_pipeline(sample_df, trained_model):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.pkl"
        save_model(trained_model, model_path)
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
    assert isinstance(loaded_model, Pipeline)


def test_save_and_load_model_predictions_match(sample_df, trained_model):
    X = sample_df[FEATURE_COLUMNS]
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.pkl"
        save_model(trained_model, model_path)
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)
    assert np.allclose(
        trained_model.predict_proba(X)[:, 1],
        loaded_model.predict_proba(X)[:, 1]
    )