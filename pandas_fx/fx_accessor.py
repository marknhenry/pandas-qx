import pandas as pd

@pd.api.extensions.register_dataframe_accessor("qx")
class QuantAccessor: 
    def __init__(self, pandas_obj):
        self._obj = pandas_obj # the underlying dataframe

    def drawdownpipe(self, colname:str, starting_point:float):
        """
        Takes a timeseries of asset returns.  
        Computes and returns a Dataframe that contains: 
        wealth index, previous peaks, and drawdown
        """

        df = self._obj.copy()

        wealth_index_col = colname+'_wealth_index'
        df[wealth_index_col] = starting_point * (1 + df[colname]).cumprod()

        previous_peaks_col = colname+'_previous_peaks'
        df[previous_peaks_col] = df[wealth_index_col].cummax()

        drawdowns_col = colname+'_drawdowns'
        df[drawdowns_col] = (df[wealth_index_col]-df[previous_peaks_col])/df[previous_peaks_col]

        return df
