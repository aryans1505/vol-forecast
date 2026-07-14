# vol-forecast

HAR forecasts of S&P 500 realized volatility, tested against random walk, EWMA
and GARCH(1,1). Daily SPY data from Jan 1994 to Jul 2026 (8,181 days),
out-of-sample from 2004. The vol proxy is Garman-Klass daily variance, with
Parkinson as a cross-check.

Main result: HAR wins at every horizon. Adding VIX helps at the one tenor it's
priced for (22 days), does nothing at 5 days, and makes 1-day forecasts worse.

## Results

QLIKE, out-of-sample 2004-2026, ~5,600 days (lower is better):

| model | 1d | 5d | 22d |
|---|---|---|---|
| rw | 0.644 | 0.311 | 0.386 |
| ewma | 0.477 | 0.323 | 0.330 |
| garch | 0.486 | 0.336 | 0.338 |
| har | **0.378** | 0.249 | 0.283 |
| har_vix | 0.748 | **0.242** | **0.258** |

Diebold-Mariano: HAR beats rw, ewma and garch at all three horizons (stats -1.9
to -15.3). har_vix vs har: -2.74 at 22d, -0.61 at 5d. Full tables in
`results/metrics.csv` and `results/dm_tests.csv`.

The 1-day har_vix number isn't a typo. The OLS weight on VIX pushes short-horizon
forecasts negative in calm markets, and the floor only partly saves it. Log-HAR
or HARQ would probably fix it; I didn't pursue that here.

## Method

- Vol proxy: Garman-Klass range-based daily variance from OHLC
  (`src/volsig/estimators.py`).
- Targets: average daily variance over the next 1, 5 and 22 trading days, never
  including the forecast date itself.
- Models (`src/volsig/models.py`):
  - `rw` — trailing h-day mean variance (a tougher baseline than lag-1)
  - `ewma` — RiskMetrics EWMA (lambda = 0.94) on close-to-close returns
  - `garch` — GARCH(1,1), parameters re-estimated every 21 days on an expanding
    window (`arch` package), then a fixed-parameter recursion so forecasts at t
    only use data through t; multi-step from the closed-form mean reversion
  - `har` — OLS of forward variance on daily, weekly (5d) and monthly (22d)
    trailing variance, fit separately per horizon
  - `har_vix` — same plus VIX, converted to daily variance units
- Walk-forward: expanding window, refit every 21 days, OOS from Jan 2004. When
  fitting at time t the training rows stop at t - h, so every training target is
  fully realized before the forecast date. There are unit tests for this
  (`tests/test_vol.py`).
- Losses: QLIKE and RMSE on annualized vol. Diebold-Mariano with Newey-West
  errors (h-1 lags) and the HLN small-sample correction.
- Negative OLS variance forecasts get floored at the 1st percentile of the
  training target (mostly hits har_vix at short horizons; counts are printed).

## Notes

- Garman-Klass only sees intraday variance, so levels sit low: median 10.2%
  annualized vs ~16% close-to-close. Model comparisons are unaffected — every
  model targets the same proxy.
- VIX ran 6.1 vol points above subsequent 22-day realized vol on average. Option
  buyers pay up for variance insurance, so VIX is a biased forecast, though
  still an informative one at its own tenor.
- One asset. A tick-level realized variance target (5-minute RV) would be
  sharper than a range proxy. And there are no transaction costs or option
  strategies anywhere in here — this measures forecast quality, nothing else.
- References: Corsi (2009) for HAR; Patton (2011) for QLIKE; Garman & Klass
  (1980) and Parkinson (1980) for the estimators.

## Running it

```
pip install -r requirements.txt
python scripts/download_data.py   # SPY + VIX daily OHLC via yfinance
python -m pytest tests -q
python scripts/run_analysis.py
```
