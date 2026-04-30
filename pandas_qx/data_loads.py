from importlib.resources import files

import pandas as pd


def get_vw_rets() -> pd.DataFrame:
    """
    Load the Value-Weighted Returns of the CRSP Index.

    Returns
    -------
    pd.DataFrame
        Monthly returns in decimal form, indexed by ``pd.PeriodIndex`` (monthly),
        with one column per industry portfolio.
    """
    data_path = files("pandas_qx").joinpath("data/ind30_m_vw_rets.csv")
    ind = pd.read_csv(data_path, header=0, index_col=0, parse_dates=True)
    ind = ind / 100
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = [c.strip() for c in ind.columns]

    return ind


def get_ffme_returns(select: str = "10") -> pd.DataFrame:
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap.

    Parameters
    ----------
    select : str, optional
        Decile cut to load. Use ``"10"`` for top/bottom 10% (default) or
        ``"20"`` for top/bottom 20%.

    Returns
    -------
    pd.DataFrame
        Monthly returns in decimal form, indexed by ``pd.PeriodIndex`` (monthly),
        with columns ``SmallCap`` and ``LargeCap``.
    """
    if select not in ("10", "20"):
        raise ValueError(f"select must be '10' or '20', got '{select}'.")
    data_path = files("pandas_qx").joinpath("data/Portfolios_Formed_on_ME_monthly_EW.csv")
    me_m = pd.read_csv(data_path, header=0, index_col=0, na_values=-99.99)
    if select == "20":
        rets = me_m[['Lo 20', 'Hi 20']]
        rets.columns = ['SmallCap', 'LargeCap']
    else:
        rets = me_m[['Lo 10', 'Hi 10']]
        rets.columns = ['SmallCap', 'LargeCap']
    rets = rets / 100
    rets.index = pd.to_datetime(rets.index, format='%Y%m').to_period('M')

    return rets


def get_hfi_returns() -> pd.DataFrame:
    """
    Load the EDHEC Hedge Fund Index returns.

    Returns
    -------
    pd.DataFrame
        Monthly returns in decimal form, indexed by ``pd.PeriodIndex`` (monthly),
        with one column per hedge fund strategy.
    """
    data_path = files("pandas_qx").joinpath("data/edhec-hedgefundindices.csv")
    hfi = pd.read_csv(data_path, header=0, index_col=0, parse_dates=True, dayfirst=True)
    hfi = hfi / 100
    hfi.index = hfi.index.to_period('M')

    return hfi
