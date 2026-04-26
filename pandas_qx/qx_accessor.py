import pandas as pd

@pd.api.extensions.register_dataframe_accessor("qx")
class QuantAccessor: 
    def __init__(self, pandas_obj):
        self._obj = pandas_obj # the underlying dataframe

    def wealth_index(self, returns_col:str, starting_point:float=100) -> pd.DataFrame:
        """
        Takes in a df, and the column name with a timeseries of asset returns.  
        Computes and returns a Dataframe that contains a wealth index
        """

        df = self._obj.copy()

        wealth_index_col = '_q_' + returns_col + '_wealth_index'
        df[wealth_index_col] = starting_point * (1 + df[returns_col]).cumprod()

        return df
    
    def drawdown(self, returns_col:str, starting_point:float=100) -> pd.DataFrame:
        """
        Takes in a df, and the column name with a timeseries of asset returns.  
        Computes and returns a Dataframe that contains a drawdown
        """

        df = self._obj.copy()

        wealth_index_col = '_q_' + returns_col + '_wealth_index_drawdown'
        df[wealth_index_col] = starting_point * (1 + df[returns_col]).cumprod()

        previous_peaks_col = '_q_' + returns_col + '_previous_peaks'
        df[previous_peaks_col] = df[wealth_index_col].cummax()

        drawdowns_col = '_q_' + returns_col+'_drawdowns'
        df[drawdowns_col] = (df[wealth_index_col]-df[previous_peaks_col])/df[previous_peaks_col]
        
        columns_to_drop = [wealth_index_col]
        df.drop(columns=columns_to_drop, inplace=True)

        return df
