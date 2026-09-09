# vol-forecast

HAR forecasts of S&P 500 realized volatility, tested against random walk, EWMA
and GARCH(1,1). Daily SPY data from Jan 1994 to Sep 2026 (8,225 days),
out-of-sample from 2004. The vol proxy is Garman-Klass daily variance, with
Parkinson as a cross-check.

Main result: log-space HAR with VIX wins at every horizon. Two of my earlier
headline claims died on closer inspection — "HAR beats GARCH everywhere" was
mostly a level-bias artifact, and "VIX only helps at 22 days" was an artifact
of fitting in levels. Details below.

## Results

QLIKE, out-of-sample 2004-2026, ~5,700 days (lower is better):

| model | 1d | 5d | 22d |
|---|---|---|---|
| rw | 0.641 | 0.310 | 0.385 |
| ewma | 0.435 | 0.299 | 0.350 |
| garch | 0.410 | 0.268 | 0.275 |
| har | 0.377 | 0.249 | 0.283 |
| har_vix | 0.743 | 0.241 | 0.258 |
| har_log | 0.367 | 0.235 | 0.274 |
| har_vix_log | **0.328** | **0.204** | **0.257** |

`har_vix_log` beats everything at all three horizons (Diebold-Mariano vs
`har_log`: -6.7 / -5.2 / -2.3). Full tables in `results/metrics.csv` and
`results/dm_tests.csv`. Two results changed my read of the problem:

- **Level calibration matters more than model choice at 22 days.** Before the
  c2c-to-GK rescale, GARCH's 22d QLIKE was 0.338 and "HAR beats GARCH" looked
  clean at every horizon. Calibrated GARCH scores 0.275 — statistically
  indistinguishable from levels-HAR (DM +0.72). Most of GARCH's apparent loss
  was level bias against the GK proxy, not worse dynamics.
- **The levels har_vix fit still blows up at 1 day** (0.743): the OLS weight on
  VIX drags short-horizon forecasts into the floor in calm markets. Fitting in
  logs fixes it outright (0.328). The levels row stays in the table because the
  failure mode is informative.

VIX in logs helps at every horizon, not just the 22-day tenor it's priced for —
the old "only at 22d" conclusion was the levels fit punishing itself at 1d.

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
  - `har_log`, `har_vix_log` — the same regressions in log variance, mapped
    back as exp(Xb + s²/2) with s² the training residual variance; positive by
    construction, so no flooring
- EWMA and GARCH are fit on close-to-close returns, so they forecast
  close-to-close variance, not the GK target. Both are rescaled to the target's
  level with an expanding mean(GK)/mean(r²) ratio estimated on past data only.
  Without this, QLIKE mostly scores their level bias (see Notes).
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
  annualized vs ~16% close-to-close. An earlier version of this repo scored raw
  EWMA/GARCH forecasts against the GK target anyway, which handicapped both by
  their overnight-share bias and flattered HAR; the level calibration above
  fixes that, and the tables reflect it.
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
pip install -e .[dev]
python scripts/download_data.py   # SPY + VIX daily OHLC via yfinance
python -m pytest -q
python scripts/run_analysis.py
```
