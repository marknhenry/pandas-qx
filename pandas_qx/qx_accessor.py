import pandas as pd


def _validate_returns_col(df: pd.DataFrame, returns_col: str) -> None:
    if returns_col not in df.columns:
        raise ValueError(f"Column '{returns_col}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[returns_col]):
        raise TypeError(f"Column '{returns_col}' must be numeric.")


def _validate_starting_point(starting_point: float) -> None:
    if starting_point <= 0:
        raise ValueError(f"starting_point must be positive, got {starting_point}.")


def wealth_index(df: pd.DataFrame, returns_col: str, starting_point: float = 100) -> pd.DataFrame:
    """
    Compute a wealth index from a periodic return series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the returns column.
    returns_col : str
        Name of the column with periodic returns (not prices).
    starting_point : float, optional
        Starting value of the wealth index. Default is 100.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with a ``_q_<returns_col>_wealth_index`` column added.
    """
    _validate_returns_col(df, returns_col)
    _validate_starting_point(starting_point)
    df = df.copy()

    wealth_index_col = '_q_' + returns_col + '_wealth_index'
    df[wealth_index_col] = starting_point * (1 + df[returns_col]).cumprod()

    return df


def drawdown(df: pd.DataFrame, returns_col: str, starting_point: float = 100) -> pd.DataFrame:
    """
    Compute drawdown for a periodic return series.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the returns column.
    returns_col : str
        Name of the column with periodic returns (not prices).
    starting_point : float, optional
        Starting value for the internal wealth index. Default is 100.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with ``_q_<returns_col>_previous_peaks`` and
        ``_q_<returns_col>_drawdowns`` columns added.
    """
    _validate_returns_col(df, returns_col)
    _validate_starting_point(starting_point)
    df = df.copy()

    wealth_index_col = '_q_' + returns_col + '_wealth_index_drawdown'
    df[wealth_index_col] = starting_point * (1 + df[returns_col]).cumprod()

    previous_peaks_col = '_q_' + returns_col + '_previous_peaks'
    df[previous_peaks_col] = df[wealth_index_col].cummax()

    drawdowns_col = '_q_' + returns_col + '_drawdowns'
    df[drawdowns_col] = (df[wealth_index_col] - df[previous_peaks_col]) / df[previous_peaks_col]

    df = df.drop(columns=[wealth_index_col])

    return df


@pd.api.extensions.register_dataframe_accessor("qx")
class QuantAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    def wealth_index(self, returns_col: str, starting_point: float = 100) -> pd.DataFrame:
        """
        Compute a wealth index from a periodic return series.

        Parameters
        ----------
        returns_col : str
            Name of the column with periodic returns (not prices).
        starting_point : float, optional
            Starting value of the wealth index. Default is 100.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with a ``_q_<returns_col>_wealth_index`` column added.
        """
        return wealth_index(self._obj, returns_col, starting_point)

    def drawdown(self, returns_col: str, starting_point: float = 100) -> pd.DataFrame:
        """
        Compute drawdown for a periodic return series.

        Parameters
        ----------
        returns_col : str
            Name of the column with periodic returns (not prices).
        starting_point : float, optional
            Starting value for the internal wealth index. Default is 100.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with previous-peaks and drawdown columns added.
        """
        return drawdown(self._obj, returns_col, starting_point)
