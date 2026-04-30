import numpy as np
import pandas as pd
import scipy.stats


@pd.api.extensions.register_dataframe_accessor("stats")
class StatsAccessor:
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    def get_returns_volatility(self, filter: str = None) -> pd.DataFrame:
        """
        Compute annualised return and volatility for each numeric column.

        Parameters
        ----------
        filter : str, optional
            If provided, only columns whose name contains this string are included.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by asset name with columns ``annualizedReturn``
            and ``annualizedVolatility``.

        Raises
        ------
        ValueError
            If the DataFrame index has no deterministic frequency.
        """
        df = self._obj.copy()

        if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            raise ValueError(
                "DataFrame index must be a DatetimeIndex or PeriodIndex to compute seasonality."
            )

        periods = self.get_seasonality_period(df)
        n_periods = df.shape[0]
        years = n_periods / periods if periods > 0 else np.nan

        cols = []
        for c in df.select_dtypes(include="number").columns:
            c_text = str(c)
            if not c_text.startswith("_q_"):
                if filter is None or filter in c_text:
                    cols.append(c)
        num = df[cols]

        out = {}
        for col in num.columns:
            x = num[col].dropna().to_numpy()

            cumulative_return = (x + 1).prod()
            annualized_return = cumulative_return ** (1 / years) - 1 if years > 0 else np.nan
            annualized_volatility = x.std(ddof=0) * np.sqrt(periods) if periods > 0 else np.nan

            col_stats = {
                "annualizedReturn": annualized_return,
                "annualizedVolatility": annualized_volatility,
            }
            out[col] = col_stats

        return pd.DataFrame(out).T

    def get_seasonality_period(self, data: pd.DataFrame | pd.Series) -> int:
        """
        Return the number of periods per year for the given time-series index.

        Parameters
        ----------
        data : pd.DataFrame or pd.Series
            Object whose index frequency is used to determine seasonality.

        Returns
        -------
        int
            Number of periods per year (e.g. 12 for monthly, 52 for weekly).

        Raises
        ------
        ValueError
            If the frequency cannot be determined or is not supported.
        """
        index = data.index
        freq = index.freq or pd.tseries.frequencies.to_offset(index.inferred_freq)

        if freq is None:
            raise ValueError("Cannot determine frequency of the index.")

        freq_type = freq.name.upper()

        if freq_type == "B":
            return 252
        elif freq_type == "D":
            return 365
        elif freq_type in ("W", "W-SUN", "W-MON"):
            return 52
        elif freq_type in ("MS", "M", "ME"):
            return 12
        elif freq_type in ("QS", "Q", "QE"):
            return 4
        elif freq_type.startswith("YS") or freq_type.startswith("YE") or freq_type in ("Y", "A", "AS"):
            return 1
        elif freq_type.startswith("H"):
            return 24
        elif freq_type.startswith("T") or freq_type.startswith("MIN"):
            return 60
        else:
            raise ValueError(f"Unsupported frequency: '{freq.name}'")

    def get_vars(self, level: float = 5) -> pd.DataFrame:
        """
        Compute Value at Risk (VaR) and related risk metrics for each numeric column.

        Parameters
        ----------
        level : float, optional
            Confidence level as a percentage (e.g. 5 means 5% VaR). Default is 5.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by asset name with columns: ``Semi-deviation``,
            ``VaR Historic``, ``VaR Gaussian``, ``VaR Cornish-Fisher``,
            ``CVaR Historic``.
        """
        if not 0 < level < 100:
            raise ValueError(f"level must be between 0 and 100, got {level}.")
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
            x = x[~np.isnan(x)]

            x_neg = x[x < 0]
            semideviation = x_neg.std(ddof=0) if len(x_neg) > 0 else np.nan

            var_historic = -np.percentile(x, level)

            z = scipy.stats.norm.ppf(level / 100)
            mu = x.mean()
            sigma = x.std(ddof=0)
            var_gaussian = -(mu + z * sigma)

            # Cornish-Fisher VaR: k is already excess kurtosis from scipy
            s = scipy.stats.skew(x)
            k = scipy.stats.kurtosis(x)  # excess kurtosis (fisher=True by default)
            z_cf = (
                z
                + (z**2 - 1) * s / 6
                + (z**3 - 3 * z) * k / 24
                - (2 * z**3 - 5 * z) * s**2 / 36
            )
            var_cornish_fisher = -(mu + z_cf * sigma)

            is_beyond = x <= -var_historic
            cvar_historic = -x[is_beyond].mean() if np.any(is_beyond) else np.nan

            col_stats = {
                "Semi-deviation": semideviation,
                "VaR Historic": var_historic,
                "VaR Gaussian": var_gaussian,
                "VaR Cornish-Fisher": var_cornish_fisher,
                "CVaR Historic": cvar_historic,
            }
            out[col] = col_stats

        return pd.DataFrame(out).T

    def get_moments(self, full: bool = False, p_level: float = 0.01) -> pd.DataFrame:
        """
        Compute distributional moments and normality test for each numeric column.

        Parameters
        ----------
        full : bool, optional
            If True, also include raw and central moments up to 4th order.
            Default is False.
        p_level : float, optional
            p-value threshold for the Jarque-Bera normality test. Default is 0.01.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by asset name with columns: ``mean``, ``var``,
            ``standard_deviation``, ``skewness``, ``kurtosis``,
            ``excess_kurtosis``, ``is_normal``. When ``full=True``, also includes
            ``m1_raw`` through ``m4_raw`` and ``m1_central`` through ``m4_central``.
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
            x = x[~np.isnan(x)]

            mu = x.mean()
            sigma = x.std(ddof=0)

            raw = {k: np.mean(x**k) for k in range(1, 5)}
            cen = {k: np.mean((x - mu) ** k) for k in range(1, 5)}
            skew = cen[3] / (sigma**3) if sigma != 0 else np.nan
            kurt = cen[4] / (sigma**4) if sigma != 0 else np.nan

            col_stats = {
                "mean": mu,
                "var": cen[2],
                "standard_deviation": sigma,
                "skewness": skew,
                "kurtosis": kurt,
                "excess_kurtosis": kurt - 3 if np.isfinite(kurt) else np.nan,
                "is_normal": scipy.stats.jarque_bera(x)[1] > p_level,
            }

            if full:
                for k in range(1, 5):
                    col_stats[f"m{k}_raw"] = raw[k]
                    col_stats[f"m{k}_central"] = cen[k]

            out[col] = col_stats

        return pd.DataFrame(out).T
