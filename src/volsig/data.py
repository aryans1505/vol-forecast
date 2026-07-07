"""Loaders for daily OHLC data (Yahoo Finance CSV dumps)."""
import pandas as pd


def load_ohlc(path):
    """Load a daily OHLC CSV (Date, open/high/low/close/... columns)."""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close"]].dropna()
    # guard against corrupt bars
    bad = (df["high"] < df["low"]) | (df["open"] <= 0) | (df["close"] <= 0)
    return df[~bad]


def load_close(path, col="close"):
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df.columns = [c.lower() for c in df.columns]
    return df[col].dropna()
