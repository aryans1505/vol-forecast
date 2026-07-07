"""Download SPY and VIX daily OHLC from Yahoo Finance into ./data."""
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "data").mkdir(exist_ok=True)

for symbol, name in [("SPY", "spy_daily"), ("^VIX", "vix_daily")]:
    df = yf.download(symbol, start="1994-01-01", auto_adjust=False, progress=False)
    df.columns = [c[0].lower() for c in df.columns]
    df.to_csv(ROOT / "data" / f"{name}.csv")
    print(f"{symbol}: {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}")
