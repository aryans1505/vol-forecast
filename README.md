# Realized Volatility Forecasting: HAR vs GARCH and Implied Vol

Does the HAR cascade beat standard volatility models out-of-sample, and does
implied volatility (VIX) add information? 32 years of daily S&P 500 (SPY) data
(Jan 1994 – Jul 2026), 22 years out-of-sample.

## Method

- **Vol proxy**: Garman–Klass (1980) range-based daily variance from OHLC —
  ~7x more efficient than squared close-to-close returns
  (`src/volsig/estimators.py`), with Parkinson as a cross-check.
- **Targets**: average daily variance over the next 1, 5 and 22 trading days,
  strictly after the forecast date (`forward_target`).
- **Models** (`src/volsig/models.py`):
  - `rw` — trailing h-day mean variance (a stronger random walk than lag-1)
  - `ewma` — RiskMetrics EWMA (λ = 0.94) on close-to-close returns
  - `garch` — GARCH(1,1), parameters re-estimated on an expanding window every
    21 days (`arch` package), forecasts from a fixed-parameter recursion using
    information through t only; multi-step via the closed-form mean reversion
  - `har` — HAR (Corsi 2009): OLS of forward variance on daily / weekly (5d) /
    monthly (22d) trailing variance, direct forecasts per horizon
  - `har_vix` — HAR plus VIX (converted to daily variance units)
- **Evaluation** (`src/volsig/evaluate.py`): expanding-window walk-forward,
  refit every 21 days, OOS from Jan 2004. Losses: QLIKE (robust to proxy noise,
  Patton 2011) and RMSE on annualized vol. Diebold–Mariano tests with
  Newey–West HAC (h−1 lags) and the HLN small-sample correction.
- **Leakage controls**: when forecasting at t, training rows are restricted to
  those whose forward target is fully realized by t (j ≤ t − h). Unit tests
  poison the boundary window and assert the fit is unaffected
  (`tests/test_vol.py`).
- Negative OLS variance forecasts are floored at the 1st percentile of the
  training target (affects `har_vix` at short horizons; counts printed).

## Results (out-of-sample, 2004–2026, ~5,600 days)

QLIKE (lower is better):

| model | 1d | 5d | 22d |
|---|---|---|---|
| rw | 0.644 | 0.311 | 0.386 |
| ewma | 0.477 | 0.323 | 0.330 |
| garch | 0.486 | 0.336 | 0.338 |
| har | **0.378** | 0.249 | 0.283 |
| har_vix | 0.748 | **0.242** | **0.258** |

Diebold–Mariano: HAR beats rw / ewma / garch at every horizon
(stats −1.9 to −15.3). Full tables: `results/metrics.csv`, `results/dm_tests.csv`.

## What the results actually say

1. **The HAR cascade wins**: three OLS coefficients on trailing vol beat
   GARCH(1,1) and EWMA everywhere, consistent with the long-memory literature.
   Persistence, not model sophistication, is what matters at these horizons.
2. **Implied vol helps exactly at the tenor it is priced for**: adding VIX
   improves 22-day forecasts (DM −2.74) — VIX is a 30-calendar-day measure —
   is a wash at 5 days (DM −0.61), and *degrades* 1-day QLIKE, where the OLS
   weight on VIX pushes forecasts negative in calm regimes (clipped at the
   floor). Information content is horizon-specific.
3. **The variance risk premium is visible**: VIX exceeds subsequent 22-day
   realized vol by 6.1 vol points on average — buyers of index options pay a
   premium for variance insurance, which is why VIX is a biased (but still
   informative) forecast.
4. Garman–Klass measures intraday variance only (no overnight gaps), so
   levels understate total vol (median 10.2% annualized vs ~16% close-to-close);
   comparisons across models are unaffected since all target the same proxy.

## Limitations (deliberately not hidden)

- One asset (SPY). Range-based proxy, not tick-level realized variance —
  5-minute RV (e.g. LOBSTER/TAQ) would sharpen the target.
- Linear HAR in variance levels; log-HAR or HARQ (Bollerslev–Patton–Quaedvlieg)
  would likely fix the short-horizon VIX pathology and is the natural next step.
- No transaction-cost or option-strategy layer: this measures forecast quality,
  not a tradeable vol strategy.

## Reproduce

```
pip install -r requirements.txt
python scripts/download_data.py   # SPY + VIX daily OHLC via yfinance
python -m pytest tests -q
python scripts/run_analysis.py
```
