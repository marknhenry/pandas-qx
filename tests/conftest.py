import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def monthly_returns_df():
    """Monthly returns DataFrame with a known date range."""
    index = pd.date_range("2020-01", periods=24, freq="MS")
    np.random.seed(42)
    return pd.DataFrame(
        {"asset_a": np.random.normal(0.01, 0.05, 24),
         "asset_b": np.random.normal(0.005, 0.03, 24)},
        index=index,
    )


@pytest.fixture
def simple_returns_df():
    """Small deterministic returns DataFrame for exact-value assertions."""
    index = pd.date_range("2021-01", periods=6, freq="MS")
    return pd.DataFrame(
        {"rets": [0.1, -0.2, 0.15, -0.05, 0.1, 0.0]},
        index=index,
    )
