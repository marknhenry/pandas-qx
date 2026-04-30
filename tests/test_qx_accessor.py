import pandas as pd
import numpy as np
import pytest
import pandas_qx
from pandas_qx import drawdown, wealth_index


# ---------------------------------------------------------------------------
# wealth_index() — standalone
# ---------------------------------------------------------------------------

class TestWealthIndexStandalone:
    def test_returns_dataframe(self, simple_returns_df):
        result = wealth_index(simple_returns_df, "rets")
        assert isinstance(result, pd.DataFrame)

    def test_output_column_present(self, simple_returns_df):
        result = wealth_index(simple_returns_df, "rets")
        assert "_q_rets_wealth_index" in result.columns

    def test_does_not_mutate_input(self, simple_returns_df):
        original_cols = list(simple_returns_df.columns)
        wealth_index(simple_returns_df, "rets")
        assert list(simple_returns_df.columns) == original_cols

    def test_starting_point_respected(self, simple_returns_df):
        result = wealth_index(simple_returns_df, "rets", starting_point=1000)
        assert pytest.approx(result["_q_rets_wealth_index"].iloc[0], rel=1e-6) == 1000 * 1.1

    def test_zero_starting_point_raises(self, simple_returns_df):
        with pytest.raises(ValueError, match="starting_point must be positive"):
            wealth_index(simple_returns_df, "rets", starting_point=0)

    def test_negative_starting_point_raises(self, simple_returns_df):
        with pytest.raises(ValueError, match="starting_point must be positive"):
            wealth_index(simple_returns_df, "rets", starting_point=-100)

    def test_missing_column_raises(self, simple_returns_df):
        with pytest.raises(ValueError, match="not found"):
            wealth_index(simple_returns_df, "nonexistent")

    def test_non_numeric_column_raises(self, simple_returns_df):
        df = simple_returns_df.copy()
        df["text"] = "a"
        with pytest.raises(TypeError, match="must be numeric"):
            wealth_index(df, "text")


# ---------------------------------------------------------------------------
# drawdown() — standalone
# ---------------------------------------------------------------------------

class TestDrawdownStandalone:
    def test_returns_dataframe(self, simple_returns_df):
        result = drawdown(simple_returns_df, "rets")
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_input(self, simple_returns_df):
        original_cols = list(simple_returns_df.columns)
        drawdown(simple_returns_df, "rets")
        assert list(simple_returns_df.columns) == original_cols

    def test_output_columns_present(self, simple_returns_df):
        result = drawdown(simple_returns_df, "rets")
        assert "_q_rets_previous_peaks" in result.columns
        assert "_q_rets_drawdowns" in result.columns

    def test_wealth_index_col_dropped(self, simple_returns_df):
        result = drawdown(simple_returns_df, "rets")
        assert "_q_rets_wealth_index_drawdown" not in result.columns

    def test_drawdown_never_positive(self, simple_returns_df):
        result = drawdown(simple_returns_df, "rets")
        assert (result["_q_rets_drawdowns"] <= 0).all()

    def test_drawdown_zero_at_peaks(self, simple_returns_df):
        result = drawdown(simple_returns_df, "rets")
        dd = result["_q_rets_drawdowns"]
        peaks = result["_q_rets_previous_peaks"]
        wealth = 100 * (1 + simple_returns_df["rets"]).cumprod()
        at_peak = wealth == peaks
        assert (dd[at_peak] == 0).all()

    def test_custom_starting_point(self, simple_returns_df):
        result_100 = drawdown(simple_returns_df, "rets", starting_point=100)
        result_1 = drawdown(simple_returns_df, "rets", starting_point=1)
        pd.testing.assert_series_equal(
            result_100["_q_rets_drawdowns"].reset_index(drop=True),
            result_1["_q_rets_drawdowns"].reset_index(drop=True),
        )

    def test_zero_starting_point_raises(self, simple_returns_df):
        with pytest.raises(ValueError, match="starting_point must be positive"):
            drawdown(simple_returns_df, "rets", starting_point=0)

    def test_missing_column_raises(self, simple_returns_df):
        with pytest.raises(ValueError, match="not found"):
            drawdown(simple_returns_df, "nonexistent_col")

    def test_non_numeric_column_raises(self, simple_returns_df):
        df = simple_returns_df.copy()
        df["text"] = "a"
        with pytest.raises(TypeError, match="must be numeric"):
            drawdown(df, "text")


# ---------------------------------------------------------------------------
# QuantAccessor.drawdown() — accessor delegates to standalone
# ---------------------------------------------------------------------------

class TestDrawdownAccessor:
    def test_accessor_matches_standalone(self, simple_returns_df):
        via_accessor = simple_returns_df.qx.drawdown("rets")
        via_standalone = drawdown(simple_returns_df, "rets")
        pd.testing.assert_frame_equal(via_accessor, via_standalone)

    def test_returns_dataframe(self, simple_returns_df):
        result = simple_returns_df.qx.drawdown("rets")
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_input(self, simple_returns_df):
        original_cols = list(simple_returns_df.columns)
        simple_returns_df.qx.drawdown("rets")
        assert list(simple_returns_df.columns) == original_cols


# ---------------------------------------------------------------------------
# QuantAccessor.wealth_index() — accessor delegates to standalone
# ---------------------------------------------------------------------------

class TestWealthIndex:
    def test_accessor_matches_standalone(self, simple_returns_df):
        via_accessor = simple_returns_df.qx.wealth_index("rets")
        via_standalone = wealth_index(simple_returns_df, "rets")
        pd.testing.assert_frame_equal(via_accessor, via_standalone)

    def test_returns_dataframe(self, simple_returns_df):
        result = simple_returns_df.qx.wealth_index("rets")
        assert isinstance(result, pd.DataFrame)

    def test_output_column_present(self, simple_returns_df):
        result = simple_returns_df.qx.wealth_index("rets")
        assert "_q_rets_wealth_index" in result.columns

    def test_does_not_mutate_input(self, simple_returns_df):
        original_cols = list(simple_returns_df.columns)
        simple_returns_df.qx.wealth_index("rets")
        assert list(simple_returns_df.columns) == original_cols

    def test_starting_point_respected(self, simple_returns_df):
        result = simple_returns_df.qx.wealth_index("rets", starting_point=1000)
        assert pytest.approx(result["_q_rets_wealth_index"].iloc[0], rel=1e-6) == 1000 * 1.1

    def test_wealth_index_monotone_for_positive_returns(self):
        index = pd.date_range("2021-01", periods=4, freq="MS")
        df = pd.DataFrame({"rets": [0.1, 0.1, 0.1, 0.1]}, index=index)
        result = df.qx.wealth_index("rets")
        wi = result["_q_rets_wealth_index"]
        assert (wi.diff().dropna() > 0).all()

    def test_custom_starting_point_scales_linearly(self, simple_returns_df):
        r1 = simple_returns_df.qx.wealth_index("rets", starting_point=100)["_q_rets_wealth_index"]
        r2 = simple_returns_df.qx.wealth_index("rets", starting_point=200)["_q_rets_wealth_index"]
        pd.testing.assert_series_equal(r2, r1 * 2, check_names=False)
