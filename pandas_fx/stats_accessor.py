# pandas_fx/stats_accessor.py
import numpy as np
import pandas as pd


@pd.api.extensions.register_dataframe_accessor("stats")
class StatsAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj
        
    def get_moments(self, full: bool = False) -> pd.DataFrame:
        """
        Compute mean, variance, skewness, kurtosis, and raw/central moments for each numeric column.
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
            mu = x.mean()
            sigma = x.std(ddof=0)

            raw = {k: np.mean(x**k) for k in range(1, 5)}
            cen = {k: np.mean((x - mu)**k) for k in range(1, 5)}
            skew = cen[3] / (sigma**3) if sigma != 0 else np.nan
            kurt = cen[4] / (sigma**4) if sigma != 0 else np.nan

            col_stats = {
                "mean": mu,
                "var": cen[2],                 # ddof=0 variance
                "standard_error": sigma / np.sqrt(len(x)) if len(x) > 0 else np.nan,
                "skewness": skew,
                "kurtosis": kurt,
                "excess_kurtosis": kurt - 3 if np.isfinite(kurt) else np.nan,
            }

            if full:
                for k in range(1, 5):
                    col_stats[f"m{k}_raw"] = raw[k]
                    col_stats[f"m{k}_central"] = cen[k]

            out[col] = col_stats
            

        return pd.DataFrame(out).T

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