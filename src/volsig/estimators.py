"""Daily variance from OHLC ranges.

Much more efficient than squared close-to-close returns, but only sees
intraday variance (no overnight gap).
"""
import numpy as np

LN2 = np.log(2.0)


def parkinson(df):
    hl = np.log(df["high"] / df["low"])
    return hl**2 / (4.0 * LN2)


def garman_klass(df):
    hl = np.log(df["high"] / df["low"])
    co = np.log(df["close"] / df["open"])
    return 0.5 * hl**2 - (2.0 * LN2 - 1.0) * co**2


def annualize_vol(daily_var):
    return np.sqrt(daily_var * 252.0) * 100.0
