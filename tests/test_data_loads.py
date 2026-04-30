import pandas as pd
import pytest
from pandas_qx import get_ffme_returns, get_hfi_returns, get_vw_rets


class TestGetVwRets:
    def test_returns_dataframe(self):
        result = get_vw_rets()
        assert isinstance(result, pd.DataFrame)

    def test_values_as_decimals(self):
        """Values should be in decimal form (divided by 100), not raw percentages."""
        result = get_vw_rets()
        # Mean monthly return well below 1 (100%) confirms decimal scaling
        assert result.abs().mean().mean() < 0.5

    def test_has_period_index(self):
        result = get_vw_rets()
        assert isinstance(result.index, pd.PeriodIndex)

    def test_no_empty_columns(self):
        result = get_vw_rets()
        assert result.shape[1] > 0

    def test_column_names_stripped(self):
        result = get_vw_rets()
        for col in result.columns:
            assert col == col.strip()


class TestGetFfmeReturns:
    def test_invalid_select_raises(self):
        with pytest.raises(ValueError, match="select must be"):
            get_ffme_returns(select="30")

    def test_returns_dataframe(self):
        result = get_ffme_returns()
        assert isinstance(result, pd.DataFrame)

    def test_default_columns(self):
        result = get_ffme_returns()
        assert list(result.columns) == ["SmallCap", "LargeCap"]

    def test_select_20_columns(self):
        result = get_ffme_returns(select="20")
        assert list(result.columns) == ["SmallCap", "LargeCap"]

    def test_values_as_decimals(self):
        """Values should be in decimal form (divided by 100), not raw percentages."""
        result = get_ffme_returns()
        # Mean monthly return well below 1 (100%) confirms decimal scaling
        assert result.abs().mean().mean() < 0.5

    def test_has_period_index(self):
        result = get_ffme_returns()
        assert isinstance(result.index, pd.PeriodIndex)


class TestGetHfiReturns:
    def test_returns_dataframe(self):
        result = get_hfi_returns()
        assert isinstance(result, pd.DataFrame)

    def test_values_as_decimals(self):
        result = get_hfi_returns()
        assert result.abs().max().max() < 1.0

    def test_has_period_index(self):
        result = get_hfi_returns()
        assert isinstance(result.index, pd.PeriodIndex)

    def test_no_empty_columns(self):
        result = get_hfi_returns()
        assert result.shape[1] > 0
