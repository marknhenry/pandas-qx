import pandas as pd
import numpy as np
import pytest
import pandas_qx  # registers accessors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(freq, periods=24, seed=42):
    """Return a DataFrame of returns with the given pandas frequency string."""
    np.random.seed(seed)
    index = pd.date_range("2020-01-01", periods=periods, freq=freq)
    return pd.DataFrame(
        {"a": np.random.normal(0.01, 0.05, periods),
         "b": np.random.normal(0.005, 0.03, periods)},
        index=index,
    )


@pytest.fixture
def monthly_df():
    return _make_df("MS", periods=60)


@pytest.fixture
def weekly_df():
    return _make_df("W", periods=104)


@pytest.fixture
def daily_df():
    return _make_df("D", periods=252)


@pytest.fixture
def annual_df():
    return _make_df("YS", periods=10)


# ---------------------------------------------------------------------------
# get_seasonality_period()
# ---------------------------------------------------------------------------

class TestGetSeasonalityPeriod:
    def test_monthly(self, monthly_df):
        assert monthly_df.stats.get_seasonality_period(monthly_df) == 12

    def test_weekly(self, weekly_df):
        assert weekly_df.stats.get_seasonality_period(weekly_df) == 52

    def test_daily(self, daily_df):
        assert daily_df.stats.get_seasonality_period(daily_df) == 365

    def test_business_daily(self):
        index = pd.bdate_range("2020-01-01", periods=252)
        df = pd.DataFrame({"a": np.random.normal(0, 0.01, 252)}, index=index)
        assert df.stats.get_seasonality_period(df) == 252

    def test_annual(self, annual_df):
        assert annual_df.stats.get_seasonality_period(annual_df) == 1

    def test_unsupported_frequency_raises(self):
        """An irregular index (no deterministic freq) should raise ValueError."""
        index = pd.to_datetime(["2020-01-01", "2020-01-04", "2020-01-10"])
        df = pd.DataFrame({"a": [1, 2, 3]}, index=index)
        with pytest.raises(ValueError):
            df.stats.get_seasonality_period(df)


# ---------------------------------------------------------------------------
# get_returns_volatility()
# ---------------------------------------------------------------------------

class TestGetReturnsVolatility:
    def test_non_time_index_raises(self):
        df = pd.DataFrame({"a": [0.01, 0.02, -0.01]})
        with pytest.raises(ValueError, match="DatetimeIndex or PeriodIndex"):
            df.stats.get_returns_volatility()

    def test_returns_dataframe(self, monthly_df):
        result = monthly_df.stats.get_returns_volatility()
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, monthly_df):
        result = monthly_df.stats.get_returns_volatility()
        assert "annualizedReturn" in result.columns
        assert "annualizedVolatility" in result.columns

    def test_row_per_asset(self, monthly_df):
        result = monthly_df.stats.get_returns_volatility()
        assert len(result) == 2  # columns "a" and "b"

    def test_volatility_positive(self, monthly_df):
        result = monthly_df.stats.get_returns_volatility()
        assert (result["annualizedVolatility"] > 0).all()

    def test_filter_parameter(self, monthly_df):
        result = monthly_df.stats.get_returns_volatility(filter="a")
        assert len(result) == 1
        assert result.index[0] == "a"

    def test_excludes_internal_q_columns(self, monthly_df):
        df = monthly_df.copy()
        df["_q_internal"] = 0.0
        result = df.stats.get_returns_volatility()
        assert "_q_internal" not in result.index

    def test_does_not_mutate_input(self, monthly_df):
        original_cols = list(monthly_df.columns)
        monthly_df.stats.get_returns_volatility()
        assert list(monthly_df.columns) == original_cols


# ---------------------------------------------------------------------------
# get_vars()
# ---------------------------------------------------------------------------

class TestGetVars:
    def test_invalid_level_raises(self, monthly_df):
        with pytest.raises(ValueError, match="level must be between"):
            monthly_df.stats.get_vars(level=0)

    def test_invalid_level_above_100_raises(self, monthly_df):
        with pytest.raises(ValueError, match="level must be between"):
            monthly_df.stats.get_vars(level=100)

    def test_returns_dataframe(self, monthly_df):
        result = monthly_df.stats.get_vars()
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, monthly_df):
        result = monthly_df.stats.get_vars()
        expected = {"Semi-deviation", "VaR Historic", "VaR Gaussian",
                    "VaR Cornish-Fisher", "CVaR Historic"}
        assert expected.issubset(set(result.columns))

    def test_row_per_asset(self, monthly_df):
        result = monthly_df.stats.get_vars()
        assert len(result) == 2

    def test_var_historic_positive(self, monthly_df):
        result = monthly_df.stats.get_vars()
        assert (result["VaR Historic"] > 0).all()

    def test_cvar_ge_var(self, monthly_df):
        """CVaR must be >= VaR Historic (CVaR is the tail mean beyond VaR)."""
        result = monthly_df.stats.get_vars()
        assert (result["CVaR Historic"] >= result["VaR Historic"]).all()

    def test_custom_level(self, monthly_df):
        result_5 = monthly_df.stats.get_vars(level=5)
        result_1 = monthly_df.stats.get_vars(level=1)
        # Stricter level → larger VaR
        assert (result_1["VaR Historic"] >= result_5["VaR Historic"]).all()

    def test_excludes_internal_q_columns(self, monthly_df):
        df = monthly_df.copy()
        df["_q_internal"] = 0.0
        result = df.stats.get_vars()
        assert "_q_internal" not in result.index

    def test_does_not_mutate_input(self, monthly_df):
        original_cols = list(monthly_df.columns)
        monthly_df.stats.get_vars()
        assert list(monthly_df.columns) == original_cols


# ---------------------------------------------------------------------------
# get_moments()
# ---------------------------------------------------------------------------

class TestGetMoments:
    def test_returns_dataframe(self, monthly_df):
        result = monthly_df.stats.get_moments()
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, monthly_df):
        result = monthly_df.stats.get_moments()
        expected = {"mean", "var", "standard_deviation", "skewness", "kurtosis",
                    "excess_kurtosis", "is_normal"}
        assert expected.issubset(set(result.columns))

    def test_row_per_asset(self, monthly_df):
        result = monthly_df.stats.get_moments()
        assert len(result) == 2

    def test_full_flag_adds_moment_columns(self, monthly_df):
        result = monthly_df.stats.get_moments(full=True)
        for k in range(1, 5):
            assert f"m{k}_raw" in result.columns
            assert f"m{k}_central" in result.columns

    def test_full_false_no_moment_columns(self, monthly_df):
        result = monthly_df.stats.get_moments(full=False)
        assert "m1_raw" not in result.columns

    def test_variance_positive(self, monthly_df):
        result = monthly_df.stats.get_moments()
        assert (result["var"] > 0).all()

    def test_is_normal_is_bool(self, monthly_df):
        result = monthly_df.stats.get_moments()
        assert result["is_normal"].dtype == bool or result["is_normal"].map(type).eq(bool).all() \
               or result["is_normal"].isin([True, False]).all()

    def test_excludes_internal_q_columns(self, monthly_df):
        df = monthly_df.copy()
        df["_q_internal"] = 0.0
        result = df.stats.get_moments()
        assert "_q_internal" not in result.index

    def test_does_not_mutate_input(self, monthly_df):
        original_cols = list(monthly_df.columns)
        monthly_df.stats.get_moments()
        assert list(monthly_df.columns) == original_cols
