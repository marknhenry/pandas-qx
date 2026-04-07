from importlib.resources import files

import pandas as pd


def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """

    data_path = files("pandas_fx").joinpath("data/Portfolios_Formed_on_ME_monthly_EW.csv")
    me_m = pd.read_csv(data_path, header=0, index_col=0, na_values=-99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format='%Y%m').to_period('M')

    return rets

def get_hfi_returns():
    """
    Load the EDHEC Hedge Fund Index returns
    """

    data_path = files("pandas_fx").joinpath("data/edhec-hedgefundindices.csv")
    hfi = pd.read_csv(data_path, header=0, index_col=0, parse_dates=True, dayfirst=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')

    return hfi

