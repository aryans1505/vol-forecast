"""Range-based daily variance estimators from OHLC bars.

Both are unbiased for the diffusion variance under zero intraday drift and
are 5-7x more efficient than squared close-to-close returns (Parkinson 1980,
Garman & Klass 1980). They measure intraday variance only (no overnight gap).
"""
import numpy as np

LN2 = np.log(2.0)


def parkinson(df):
    """Parkinson (1980) high-low variance estimator, per day."""
    hl = np.log(df["high"] / df["low"])
    return hl**2 / (4.0 * LN2)


def garman_klass(df):
    """Garman-Klass (1980) OHLC variance estimator, per day."""
    hl = np.log(df["high"] / df["low"])
    co = np.log(df["close"] / df["open"])
    return 0.5 * hl**2 - (2.0 * LN2 - 1.0) * co**2


def annualize_vol(daily_var):
    """Daily variance -> annualized vol in %."""
    return np.sqrt(daily_var * 252.0) * 100.0
