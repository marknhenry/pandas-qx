import numpy as np
import pandas as pd
import scipy.stats
# from scipy.stats import norm


@pd.api.extensions.register_dataframe_accessor("stats")
class StatsAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj
        
    def get_vars(self, level: float = 5) -> pd.DataFrame:
        """
        Compute the Value At Risk (VaR) using historical, Gaussian, and Cornish-Fisher methods for each numeric column. 
        Returns a DataFrame with these statistics.
        """
        df = self._obj.copy()
        
        cols = []
        for c in df.select_dtypes(include="number").columns:
            c_text = str(c)
            if not c_text.startswith("_q_"):
                cols.append(c)
        num = df[cols]

        out = {}
        for col in num.columns:
            x = num[col].dropna().to_numpy()
            x = x[~np.isnan(x)]  # Remove any NaNs that might be present
            
            is_negative = x < 0
            x_neg = x[is_negative]
            semideviation = x_neg.std(ddof=0) if len(x_neg) > 0 else np.nan
            
            # Historical VaR is the negative of the quantile at the given level
            var_historic = -np.percentile(x, level)
            
            # Gaussian VaR is - (mean + z * std) where z is the z-score corresponding to the confidence level
            z = scipy.stats.norm.ppf(level / 100)
            mu = x.mean()
            sigma = x.std(ddof=0)
            var_gaussian = -(mu + z * sigma)
            
            # Cornish-Fisher VaR adjusts the z-score for skewness and kurtosis
            s = scipy.stats.skew(x)
            k = scipy.stats.kurtosis(x)  # excess kurtosis
            z_cf = (z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * (k-3) / 24 - (2*z**3 - 5*z) * s**2 / 36)
            var_cornish_fisher = -(mu + z_cf * sigma)
            
            # CVaR is the average of 
            is_beyond = x <= -var_historic
            cvar_historic = -x[is_beyond].mean() if np.any(is_beyond) else np.nan
            
            col_stats = {
                "Semi-deviation": semideviation,
                "VaR Historic": var_historic,
                "VaR Gaussian": var_gaussian,
                "VaR Cornish-Fisher": var_cornish_fisher, 
                "CVaR Historic": cvar_historic
            }
            out[col] = col_stats
            

        return pd.DataFrame(out).T
        
    def get_moments(self, full: bool = False, p_level: float = 0.01) -> pd.DataFrame:
        """
        Compute mean, variance, skewness, kurtosis, and raw/central moments for each numeric column.
        Returns a DataFrame with these statistics.
        It will check for normaility using the Jarque-Bera test using the given p-value threshold. If full=True, it also includes raw and central moments up to 4th order.
        # If negative_only=True, it will filter out any non-negative values.  
        """
        df = self._obj.copy()
        
        
        cols = []
        for c in df.select_dtypes(include="number").columns:
            c_text = str(c)
            if not c_text.startswith("_q_"):
                cols.append(c)
        num = df[cols]

        out = {}
        for col in num.columns:
            x = num[col].dropna().to_numpy()
            x = x[~np.isnan(x)]  # Remove any NaNs that might be present
            
            mu = x.mean()
            sigma = x.std(ddof=0)

            raw = {k: np.mean(x**k) for k in range(1, 5)}
            cen = {k: np.mean((x - mu)**k) for k in range(1, 5)}
            skew = cen[3] / (sigma**3) if sigma != 0 else np.nan
            kurt = cen[4] / (sigma**4) if sigma != 0 else np.nan
            
            # x_neg = x[x < 0]
            # semideviation = np.sqrt(np.mean((x_neg - x_neg.mean())**2)) if len(x_neg) > 0 else np.nan

            col_stats = {
                "mean": mu,
                "var": cen[2],                 # ddof=0 variance
                "standard_deviation": sigma / np.sqrt(len(x)) if len(x) > 0 else np.nan,
                # "semideviation": semideviation,
                "skewness": skew,
                "kurtosis": kurt,
                "excess_kurtosis": kurt - 3 if np.isfinite(kurt) else np.nan,
                "is_normal": scipy.stats.jarque_bera(x)[1] > p_level
            }

            if full:
                for k in range(1, 5):
                    col_stats[f"m{k}_raw"] = raw[k]
                    col_stats[f"m{k}_central"] = cen[k]

            out[col] = col_stats
            

        return pd.DataFrame(out).T
    
    # def get_vars(self, full: bool = False, p_level: float = 0.01) -> pd.DataFrame:
    #     """
        
    #     """
        
    #     return;
    
    # def var_historic(r, level=5):
    #     """
    #     Compute the historical Value at Risk (VaR) at the given confidence level.
    #     VaR is the negative of the quantile of the returns distribution corresponding to the confidence level.
    #     For example, var_historic(r, level=5) returns the 5% VaR, which is -np.percentile(r, 5).
    #     """
    #     if isinstance(r, pd.DataFrame):
    #         return r.aggregate(var_historic, level=level)
    #     elif isinstance(r, pd.Series):
    #         return -np.percentile(r, level)
    #     else: 
    #         raise TypeError("Input must be a pandas Series or DataFrame.")
    
    # def var_gaussian(r, level=5):
    #     """
    #     Compute the Gaussian (parametric) Value at Risk (VaR) at the given confidence level.
    #     VaR is calculated as - (mean + z * std), where z is the z-score corresponding to the confidence level.
    #     For example, var_gaussian(r, level=5) returns the 5% VaR, which is - (mean + z * std) where z = scipy.stats.norm.ppf(0.05).
    #     """
    #     if isinstance(r, pd.DataFrame):
    #         return r.aggregate(var_gaussian, level=level)
    #     elif isinstance(r, pd.Series):
    #         mu = r.mean()
    #         sigma = r.std(ddof=0)
    #         z = scipy.stats.norm.ppf(level / 100)
    #         return -(mu + z * sigma)
    #     else: 
    #         raise TypeError("Input must be a pandas Series or DataFrame.")
    
    
    # def is_normal(self, r, p_value=0.01):
    #     """
    #     Applies the Jacque-Bera test for normality to a Series. Returns True if the null hypothesis of 
    #     normality is not rejected at the given significance level.
    #     """
    #     from scipy.stats import normaltest
    #     stat, p = jarque_bera(r) 
    #     return p > p_value

    # def normalised_mean(
    #     self,
    #     cols=None,
    #     method: str = "minmax",
    #     numeric_only: bool = True,
    #     skipna: bool = True,
    # ) -> pd.Series:
    #     """
    #     Normalise each selected column to [-1, 1], then compute the mean of each column.

    #     method:
    #       - "minmax": -1 + 2*(x - min)/(max - min)
    #       - "maxabs": x / max(abs(x))  (already in [-1,1] unless all zeros)
    #       - "zscore_clip": z-score then clip to [-1,1] (less common but robust-ish)
    #     """
    #     df = self._obj

    #     # Select columns
    #     if cols is None:
    #         if numeric_only:
    #             data = df.select_dtypes(include="number")
    #         else:
    #             data = df
    #     else:
    #         data = df[cols]

    #     if data.shape[1] == 0:
    #         return pd.Series(dtype=float)

    #     # Normalise per column
    #     if method == "minmax":
    #         col_min = data.min(axis=0, skipna=skipna)
    #         col_max = data.max(axis=0, skipna=skipna)
    #         denom = (col_max - col_min).replace(0, np.nan)
    #         normed = -1 + 2 * (data - col_min) / denom

    #     elif method == "maxabs":
    #         scale = data.abs().max(axis=0, skipna=skipna).replace(0, np.nan)
    #         normed = data / scale

    #     elif method == "zscore_clip":
    #         mu = data.mean(axis=0, skipna=skipna)
    #         sigma = data.std(axis=0, ddof=0, skipna=skipna).replace(0, np.nan)
    #         z = (data - mu) / sigma
    #         normed = z.clip(-1, 1)

    #     else:
    #         raise ValueError(f"Unknown method: {method}")

    #     # Column-wise mean of normalised data
    #     return normed.mean(axis=0, skipna=skipna)